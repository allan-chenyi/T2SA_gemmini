package gemmini

import chisel3._
import chisel3.util._

import gemmini.Util._

/**
 * TwistMeshWithDelays: drop-in replacement for MeshWithDelays using Twist-WS mesh.
 *
 * Same external IO and protocol as MeshWithDelays. Internally:
 *   - Maps d input → in_weight (weight data)
 *   - Maps a input → in_ifmap (activation data)
 *   - Maps b input → in_psum (bias data, sign-extended to outputType)
 *   - Generates staggered lock signal from fire_counter
 *   - Passes propagate for double-buffer control
 *
 * ExecuteController feeds D (weight) rows in forward order for TwistMesh
 * (d_address = base + counter), so lock fires in matching forward order.
 *
 * After each request's n-cycle firing, a drain phase of n-1 cycles keeps
 * mesh.io.in_valid=true (with weight=0, lock=false) so that lock/weight
 * signals finish propagating through the pass_reg chain to all PEs.
 * Lock propagation and pass_reg both freeze during stalls (gated by in_valid).
 *
 * @param dualOp  false = single-op (A*B only, LP1 fixed), true = dual-op (A*B and A*B^T)
 */
class TwistMeshWithDelays[T <: Data: Arithmetic, U <: TagQueueTag with Data]
  (val inputType: T, val weightType: T, val outputType: T, accType: T,
   tagType: U, df: Dataflow.Value, tree_reduction: Boolean, tile_latency: Int, output_delay: Int,
   tileRows: Int, tileColumns: Int, meshRows: Int, meshColumns: Int,
   leftBanks: Int, upBanks: Int, outBanks: Int = 1, n_simultaneous_matmuls: Int = -1,
   dualOp: Boolean = false)
  extends Module {

  val ev = implicitly[Arithmetic[T]]
  import ev._

  val A_TYPE = Vec(meshRows, Vec(tileRows, inputType))
  val B_TYPE = Vec(meshColumns, Vec(tileColumns, weightType))
  val C_TYPE = Vec(meshColumns, Vec(tileColumns, outputType))
  val D_TYPE = Vec(meshColumns, Vec(tileColumns, weightType))

  assert(meshRows * tileRows == meshColumns * tileColumns)
  val block_size = meshRows * tileRows
  val n = block_size // Twist mesh dimension (square)

  val latency_per_pe = ((tile_latency + 1).toFloat / (tileRows min tileColumns)) max 1.0f
  val max_simultaneous_matmuls = if (n_simultaneous_matmuls == -1) {
    (5 * latency_per_pe).ceil.toInt
  } else {
    n_simultaneous_matmuls
  }

  val tagqlen = max_simultaneous_matmuls + 1

  // ========================================================================
  // External IO — identical to MeshWithDelays
  // ========================================================================
  val io = IO(new Bundle {
    val a = Flipped(Decoupled(A_TYPE))
    val b = Flipped(Decoupled(B_TYPE))
    val d = Flipped(Decoupled(D_TYPE))

    val req = Flipped(Decoupled(new MeshWithDelaysReq(accType, tagType.cloneType, block_size)))

    val resp = Valid(new MeshWithDelaysResp(outputType, meshColumns, tileColumns, block_size, tagType.cloneType))

    val tags_in_progress = Output(Vec(tagqlen, tagType))
  })

  // ========================================================================
  // Internal TwistMesh
  // ========================================================================
  val mesh = Module(new TwistMesh(n, dualOp, inputType, weightType, outputType, accType, max_simultaneous_matmuls))

  // ========================================================================
  // Request handling (same as MeshWithDelays — no state machine)
  // ========================================================================
  val req = Reg(UDValid(new MeshWithDelaysReq(accType, tagType, block_size)))

  val matmul_id = RegInit(0.U(log2Up(max_simultaneous_matmuls).W))
  val in_prop = RegInit(0.U(1.W))

  val total_fires = req.bits.total_rows
  val fire_counter = RegInit(0.U(log2Up(block_size).W))

  val a_buf = RegEnable(io.a.bits, io.a.fire)
  val b_buf = RegEnable(io.b.bits, io.b.fire)
  val d_buf = RegEnable(io.d.bits, io.d.fire)

  val a_written = RegInit(false.B)
  val b_written = RegInit(false.B)
  val d_written = RegInit(false.B)

  val input_next_row = req.valid && ((a_written && b_written && d_written) || req.bits.flush > 0.U)
  val last_fire = fire_counter === total_fires - 1.U && input_next_row

  when(io.req.fire) {
    req.push(io.req.bits)
    in_prop := io.req.bits.pe_control.propagate ^ in_prop
    matmul_id := wrappingAdd(matmul_id, 1.U, max_simultaneous_matmuls)
  }.elsewhen(last_fire) {
    req.valid := req.bits.flush > 1.U
    req.bits.flush := req.bits.flush - 1.U
  }

  when(input_next_row) {
    a_written := false.B
    b_written := false.B
    d_written := false.B
    fire_counter := wrappingAdd(fire_counter, 1.U, total_fires)
  }

  when(io.a.fire) { a_written := true.B }
  when(io.b.fire) { b_written := true.B }
  when(io.d.fire) { d_written := true.B }

  io.a.ready := !a_written || input_next_row || io.req.ready
  io.b.ready := !b_written || input_next_row || io.req.ready
  io.d.ready := !d_written || input_next_row || io.req.ready

  assert(req.valid || !input_next_row)

  val pause = !req.valid || !input_next_row

  // ========================================================================
  // Transposer (same as MeshWithDelays)
  // ========================================================================
  val a_is_from_transposer = Mux(req.bits.pe_control.dataflow === Dataflow.OS.id.U,
    !req.bits.a_transpose, req.bits.a_transpose)
  val d_is_from_transposer = req.bits.pe_control.dataflow === Dataflow.WS.id.U && req.bits.bd_transpose
  val b_is_from_transposer = req.bits.pe_control.dataflow === Dataflow.OS.id.U && req.bits.bd_transpose

  val transposer = Module(new AlwaysOutTransposer(block_size, inputType))

  transposer.io.inRow.valid := !pause && (a_is_from_transposer || b_is_from_transposer || d_is_from_transposer)
  transposer.io.inRow.bits := MuxCase(VecInit(a_buf.flatten), Seq(
    b_is_from_transposer -> VecInit(b_buf.flatten),
    d_is_from_transposer -> VecInit(d_buf.flatten.reverse),
  ))
  transposer.io.outCol.ready := true.B
  val transposer_out = VecInit(transposer.io.outCol.bits.grouped(tileRows).map(t => VecInit(t)).toSeq)

  // Resolved input data (after optional transpose)
  val a_data = Mux(a_is_from_transposer, transposer_out.asTypeOf(A_TYPE), a_buf)
  val d_data = Mux(d_is_from_transposer,
    VecInit(transposer_out.flatten.reverse.grouped(tileRows).map(VecInit(_)).toSeq).asTypeOf(D_TYPE), d_buf)

  // ========================================================================
  // Signal mapping to TwistMesh
  // ========================================================================

  // Weight: d data during normal operation, 0 during drain
  for (r <- 0 until n) {
    mesh.io.in_weight(r) := d_data(r)(0).asTypeOf(weightType)
  }

  // Lock: staggered in forward order matching ExecuteController's forward D read
  // (for TwistMesh, ExecuteController sends D in forward order: row 0, 1, ..., n-1).
  // fire_counter=0 → lock row 0, fire_counter=1 → lock row 1, etc.
  // During drain, lock=false but already-sent locks continue propagating
  // through the pass_reg/lock_delayed chain (since in_valid stays true).
  for (r <- 0 until n) {
    mesh.io.in_lock(r) := !pause && (r.U === fire_counter)
  }

  // Ifmap: a data during normal operation, 0 during drain
  for (r <- 0 until n) {
    mesh.io.in_ifmap(r) := a_data(r)(0)
  }

  // Psum: B data as bias during normal operation, 0 during drain
  // In WS mode, B port carries the bias (from COMPUTE's rs2 address).
  // Each row r gets b_buf(r)(0) cast to outputType as initial psum.
  for (r <- 0 until n) {
    mesh.io.in_psum(r) := b_buf(r)(0).withWidthOf(outputType)
  }

  // Control
  mesh.io.in_propagate := in_prop
  mesh.io.in_valid := !pause
  mesh.io.in_last  := last_fire
  mesh.io.in_id    := matmul_id

  // Dual-op: set load pattern from request (bd_transpose selects LP2 for A*B^T)
  if (dualOp) {
    mesh.io.load_pattern.get := Mux(req.bits.bd_transpose, 1.U, 0.U)
  }

  // ========================================================================
  // Response — valid/last propagate through mesh, naturally aligned with data
  // ========================================================================
  val resp_data = Reg(Vec(meshColumns, Vec(tileColumns, outputType)))
  for (r <- 0 until n) {
    resp_data(r)(0) := mesh.io.out_psum(r)
  }

  io.resp.valid := RegNext(mesh.io.out_valid, false.B)
  io.resp.bits.data := resp_data
  io.resp.bits.last := RegNext(mesh.io.out_last, false.B)

  // ========================================================================
  // Tag queue (same structure as MeshWithDelays)
  // ========================================================================
  class TagWithIdAndTotalRows extends Bundle with TagQueueTag {
    val tag = tagType.cloneType
    val id = UInt(log2Up(max_simultaneous_matmuls).W)
    val total_rows = UInt(log2Up(block_size + 1).W)

    override def make_this_garbage(dummy: Int = 0): Unit = {
      total_rows := block_size.U
      tag.make_this_garbage()
    }
  }

  // matmul_id_of_output = matmul_id + 2: PRELOAD's tag is designed to match
  // the COMPUTE output (one matmul later). Same as standard MeshWithDelays.
  val matmul_id_of_output = wrappingAdd(matmul_id,
    Mux(io.req.bits.pe_control.dataflow === Dataflow.OS.id.U, 3.U, 2.U),
    max_simultaneous_matmuls)
  val matmul_id_of_current = wrappingAdd(matmul_id, 1.U, max_simultaneous_matmuls)

  val tagq = Module(new TagQueue(new TagWithIdAndTotalRows, tagqlen))
  tagq.io.enq.valid := io.req.fire && io.req.bits.flush === 0.U
  tagq.io.enq.bits.tag := io.req.bits.tag
  tagq.io.enq.bits.total_rows := DontCare
  tagq.io.enq.bits.id := matmul_id_of_output

  val tag_garbage = Wire(tagType.cloneType)
  tag_garbage := DontCare
  tag_garbage.make_this_garbage()

  // matmul_id propagates through mesh with data, read from mesh output.
  // RegNext to align with resp_data (1 extra cycle for resp_data register).
  val out_matmul_id = RegNext(mesh.io.out_id)

  io.resp.bits.tag := Mux(tagq.io.deq.valid && out_matmul_id === tagq.io.deq.bits.id,
    tagq.io.deq.bits.tag, tag_garbage)

  tagq.io.deq.ready := io.resp.valid && io.resp.bits.last && out_matmul_id === tagq.io.deq.bits.id

  // Total rows queue (same as MeshWithDelays)
  val total_rows_q = Module(new Queue(new TagWithIdAndTotalRows, tagqlen))
  total_rows_q.io.enq.valid := io.req.fire && io.req.bits.flush === 0.U
  total_rows_q.io.enq.bits.tag := DontCare
  total_rows_q.io.enq.bits.total_rows := io.req.bits.total_rows
  total_rows_q.io.enq.bits.id := matmul_id_of_current

  io.resp.bits.total_rows := Mux(total_rows_q.io.deq.valid && out_matmul_id === total_rows_q.io.deq.bits.id,
    total_rows_q.io.deq.bits.total_rows, block_size.U)

  total_rows_q.io.deq.ready := io.resp.valid && io.resp.bits.last && out_matmul_id === total_rows_q.io.deq.bits.id

  // ========================================================================
  // Request ready and tags in progress (same as MeshWithDelays)
  // ========================================================================
  io.req.ready := (!req.valid || last_fire) && tagq.io.enq.ready && total_rows_q.io.enq.ready
  io.tags_in_progress := tagq.io.all.map(_.tag)

  when(reset.asBool) {
    req.valid := false.B
  }

  // ========================================================================
  // Debug trace (printf for Verilator waveform analysis)
  // ========================================================================
  val dbg_cycle = RegInit(0.U(32.W))
  dbg_cycle := dbg_cycle + 1.U

  when(io.req.fire) {
    printf("[TWIST %d] REQ fire: prop=%d, in_prop=%d->%d, matmul_id=%d->%d, total_rows=%d, flush=%d\n",
      dbg_cycle,
      io.req.bits.pe_control.propagate, in_prop, io.req.bits.pe_control.propagate ^ in_prop,
      matmul_id, wrappingAdd(matmul_id, 1.U, max_simultaneous_matmuls),
      io.req.bits.total_rows, io.req.bits.flush)
  }

  when(!pause) {
    printf("[TWIST %d] FIRE fc=%d prop=%d wt0=%d wt1=%d lk0=%d lk1=%d a0=%d a1=%d b0=%d\n",
      dbg_cycle, fire_counter, in_prop,
      mesh.io.in_weight(0).asUInt, mesh.io.in_weight(1).asUInt,
      mesh.io.in_lock(0), mesh.io.in_lock(1),
      mesh.io.in_ifmap(0).asUInt, mesh.io.in_ifmap(1).asUInt,
      mesh.io.in_psum(0).asUInt)
  }

  when(io.resp.valid) {
    printf("[TWIST %d] RESP valid=%d last=%d out_id=%d tagq_id=%d tagq_valid=%d match=%d d0=%d d1=%d d2=%d d3=%d\n",
      dbg_cycle, io.resp.valid, io.resp.bits.last,
      out_matmul_id,
      tagq.io.deq.bits.id, tagq.io.deq.valid,
      tagq.io.deq.valid && out_matmul_id === tagq.io.deq.bits.id,
      io.resp.bits.data(0)(0).asUInt, io.resp.bits.data(1)(0).asUInt,
      io.resp.bits.data(2)(0).asUInt, io.resp.bits.data(3)(0).asUInt)
  }
}
