package gemmini

import chisel3._
import chisel3.util._

/**
 * Wrapper that conditionally instantiates either MeshWithDelays (standard)
 * or TwistMeshWithDelays (Twist-WS) and exposes a unified IO.
 *
 * This allows ExecuteController to use one mesh variable regardless of mesh type.
 */
class MeshWithDelaysWrapper[T <: Data: Arithmetic, U <: TagQueueTag with Data]
  (inputType: T, weightType: T, outputType: T, accType: T,
   tagType: U, df: Dataflow.Value, tree_reduction: Boolean, tile_latency: Int, output_delay: Int,
   tileRows: Int, tileColumns: Int, meshRows: Int, meshColumns: Int,
   leftBanks: Int, upBanks: Int, outBanks: Int = 1, n_simultaneous_matmuls: Int = -1,
   meshType: MeshType = StandardMesh)
  extends Module {

  val A_TYPE = Vec(meshRows, Vec(tileRows, inputType))
  val B_TYPE = Vec(meshColumns, Vec(tileColumns, weightType))
  val D_TYPE = Vec(meshColumns, Vec(tileColumns, weightType))

  assert(meshRows * tileRows == meshColumns * tileColumns)
  val block_size = meshRows * tileRows

  // Match the latency calculation from MeshWithDelays
  val latency_per_pe = ((tile_latency + 1).toFloat / (tileRows min tileColumns)) max 1.0f
  val max_simultaneous_matmuls = if (n_simultaneous_matmuls == -1) {
    (5 * latency_per_pe).ceil.toInt
  } else {
    n_simultaneous_matmuls
  }
  val tagqlen = max_simultaneous_matmuls + 1

  val io = IO(new Bundle {
    val a = Flipped(Decoupled(A_TYPE))
    val b = Flipped(Decoupled(B_TYPE))
    val d = Flipped(Decoupled(D_TYPE))

    val req = Flipped(Decoupled(new MeshWithDelaysReq(accType, tagType.cloneType, block_size)))

    val resp = Valid(new MeshWithDelaysResp(outputType, meshColumns, tileColumns, block_size, tagType.cloneType))

    val tags_in_progress = Output(Vec(tagqlen, tagType))
  })

  meshType match {
    case StandardMesh => {
      val inner = Module(new MeshWithDelays(inputType, weightType, outputType, accType,
        tagType, df, tree_reduction, tile_latency, output_delay,
        tileRows, tileColumns, meshRows, meshColumns,
        leftBanks, upBanks, outBanks, n_simultaneous_matmuls))

      inner.io.a <> io.a
      inner.io.b <> io.b
      inner.io.d <> io.d
      inner.io.req <> io.req
      io.resp := inner.io.resp
      io.tags_in_progress := inner.io.tags_in_progress
    }

    case TwistSingleOp | TwistDualOp => {
      val isDual = meshType == TwistDualOp
      val inner = Module(new TwistMeshWithDelays(inputType, weightType, outputType, accType,
        tagType, df, tree_reduction, tile_latency, output_delay,
        tileRows, tileColumns, meshRows, meshColumns,
        leftBanks, upBanks, outBanks, n_simultaneous_matmuls,
        dualOp = isDual))

      inner.io.a <> io.a
      inner.io.b <> io.b
      inner.io.d <> io.d
      inner.io.req <> io.req
      io.resp := inner.io.resp
      io.tags_in_progress := inner.io.tags_in_progress
    }
  }
}
