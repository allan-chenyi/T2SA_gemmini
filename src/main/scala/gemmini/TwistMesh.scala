package gemmini

import chisel3._
import chisel3.util._

/**
 * Twist-WS Mesh: an n*n array of TwistPE.
 *
 * Two interconnect lines:
 *   Black (horizontal): PE(r, c) -> PE(r, c+1)
 *   Red  (diagonal):    PE(r, c) -> PE((r-1+n)%n, c+1)
 *
 * Compute pattern is always CP1:
 *   ifmap on RED (diagonal), psum on BLACK (horizontal)
 *
 * Load pattern:
 *   LP1: weight on BLACK, lock on RED
 *   LP2: weight on RED,   lock on BLACK
 *
 * All signals propagate through PE chain with 1-cycle delay per column:
 *   - weight, lock, propagate: registered inside TwistPE
 *   - ifmap, psum, valid, last: registered between PEs in this module
 *
 * When dualOp=false: LP1 is hardwired (single-op, supports A*B with row-major B).
 * When dualOp=true:  LP is runtime-selectable via io.load_pattern (dual-op).
 */
class TwistMesh[T <: Data](n: Int, dualOp: Boolean,
                            inputType: T, weightType: T, outputType: T, accType: T,
                            max_simultaneous_matmuls: Int = 1)
                           (implicit ev: Arithmetic[T]) extends Module {
  import ev._

  val io = IO(new Bundle {
    val in_weight    = Input(Vec(n, weightType))
    val in_lock      = Input(Vec(n, Bool()))
    val in_ifmap     = Input(Vec(n, inputType))
    val in_psum      = Input(Vec(n, outputType))
    val in_propagate = Input(UInt(1.W))
    val in_valid     = Input(Bool())
    val in_last      = Input(Bool())
    val in_id        = Input(UInt(log2Up(max_simultaneous_matmuls).W))

    // Runtime load pattern selection (only used when dualOp=true)
    val load_pattern = if (dualOp) Some(Input(UInt(1.W))) else None

    val out_psum     = Output(Vec(n, outputType))
    val out_valid    = Output(Bool())
    val out_last     = Output(Bool())
    val out_id       = Output(UInt(log2Up(max_simultaneous_matmuls).W))

    // Debug ports for testing
    val debug_buffer1 = Output(Vec(n, Vec(n, inputType)))
    val debug_buffer2 = Output(Vec(n, Vec(n, inputType)))
    val debug_pe_out_psum = Output(Vec(n, Vec(n, outputType)))
    val debug_pe_in_ifmap = Output(Vec(n, Vec(n, inputType)))
    val debug_pe_prop_delayed = Output(Vec(n, Vec(n, UInt(1.W))))
    val debug_pe_weight_sel = Output(Vec(n, Vec(n, inputType)))
  })

  val pes = Seq.fill(n, n)(Module(new TwistPE(inputType, weightType, outputType, accType)))

  // Helper: is LP1? (weight horizontal, lock diagonal)
  val isLP1 = if (dualOp) io.load_pattern.get === 0.U else true.B

  // --- Valid: pipelined horizontally (same as psum on BLACK) ---
  // Column 0 gets in_valid directly; subsequent columns via RegNext.
  val col_valid = Wire(Vec(n, Bool()))
  col_valid(0) := io.in_valid
  for (c <- 1 until n) {
    col_valid(c) := RegNext(col_valid(c - 1), false.B)
  }
  for (r <- 0 until n; c <- 0 until n) {
    pes(r)(c).io.in_valid := col_valid(c)
  }

  // --- Last: pipelined horizontally (same as valid) ---
  val col_last = Wire(Vec(n, Bool()))
  col_last(0) := io.in_last
  for (c <- 1 until n) {
    col_last(c) := RegNext(col_last(c - 1), false.B)
  }

  // --- Id: pipelined horizontally (same as valid/last) ---
  val col_id = Wire(Vec(n, UInt(log2Up(max_simultaneous_matmuls).W)))
  col_id(0) := io.in_id
  for (c <- 1 until n) {
    col_id(c) := RegNext(col_id(c - 1))
  }

  // --- Weight routing ---
  for (r <- 0 until n) pes(r)(0).io.in_weight := io.in_weight(r)
  for (r <- 0 until n; c <- 1 until n) {
    val horiz_src = pes(r)(c - 1).io.out_weight
    val diag_src  = pes((r + 1) % n)(c - 1).io.out_weight
    if (dualOp) {
      pes(r)(c).io.in_weight := Mux(isLP1, horiz_src, diag_src)
    } else {
      pes(r)(c).io.in_weight := horiz_src  // LP1 fixed
    }
  }

  // --- Lock routing (opposite of weight) ---
  for (r <- 0 until n) pes(r)(0).io.in_lock := io.in_lock(r)
  for (r <- 0 until n; c <- 1 until n) {
    val horiz_src = pes(r)(c - 1).io.out_lock
    val diag_src  = pes((r + 1) % n)(c - 1).io.out_lock
    if (dualOp) {
      pes(r)(c).io.in_lock := Mux(isLP1, diag_src, horiz_src)
    } else {
      pes(r)(c).io.in_lock := diag_src  // LP1 fixed: lock on RED
    }
  }

  // --- Ifmap routing: always RED (diagonal) for CP1 ---
  // Pipeline register between PEs, gated by source column's valid.
  for (r <- 0 until n) pes(r)(0).io.in_ifmap := io.in_ifmap(r)
  for (r <- 0 until n; c <- 1 until n) {
    pes(r)(c).io.in_ifmap := RegEnable(pes((r + 1) % n)(c - 1).io.out_ifmap, col_valid(c - 1))
  }

  // --- Psum routing: always BLACK (horizontal) for CP1 ---
  // Pipeline register between PEs, gated by source column's valid.
  for (r <- 0 until n) pes(r)(0).io.in_psum := io.in_psum(r)
  for (r <- 0 until n; c <- 1 until n) {
    pes(r)(c).io.in_psum := RegEnable(pes(r)(c - 1).io.out_psum, col_valid(c - 1))
  }

  // --- Propagate: pipelined left-to-right through PE chain (horizontal) ---
  for (r <- 0 until n) pes(r)(0).io.in_propagate := io.in_propagate
  for (r <- 0 until n; c <- 1 until n) {
    pes(r)(c).io.in_propagate := pes(r)(c - 1).io.out_propagate
  }

  // --- Output: from column n-1 ---
  for (r <- 0 until n) io.out_psum(r) := pes(r)(n - 1).io.out_psum
  io.out_valid := col_valid(n - 1)
  io.out_last  := col_last(n - 1)
  io.out_id    := col_id(n - 1)

  // --- Debug ---
  for (r <- 0 until n; c <- 0 until n) {
    io.debug_buffer1(r)(c) := pes(r)(c).io.debug_buffer1
    io.debug_buffer2(r)(c) := pes(r)(c).io.debug_buffer2
    io.debug_pe_out_psum(r)(c) := pes(r)(c).io.out_psum
    io.debug_pe_in_ifmap(r)(c) := pes(r)(c).io.in_ifmap
    io.debug_pe_prop_delayed(r)(c) := pes(r)(c).io.debug_prop_delayed
    io.debug_pe_weight_sel(r)(c) := pes(r)(c).io.debug_weight_sel
  }
}
