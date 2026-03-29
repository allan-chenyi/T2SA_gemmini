package gemmini

import chisel3._
import chisel3.util._

/**
 * Twist-WS Processing Element (production version).
 *
 * Three internal registers:
 *   - buffer1, buffer2: weight double-buffering (selected by propagate)
 *   - pass_reg: weight pass-through register for weight loading
 *
 * Double-buffer convention (matches standard Gemmini PE):
 *   propagate=0: compute with buffer1, load into buffer2 (inactive)
 *   propagate=1: compute with buffer2, load into buffer1 (inactive)
 *   Load and compute always use DIFFERENT buffers (true double-buffering).
 *
 * Weight loading:
 *   Weight enters from the left and flows through pass_reg (1-cycle delay, frozen during stall).
 *   Lock propagates with 1-cycle delay (frozen during stall).
 *   On lock rising edge, pass_reg is captured into the inactive buffer
 *   one cycle later (using pipelined propagate for buffer selection).
 *
 * Propagate:
 *   Propagate flows left-to-right through PE chain (1-cycle delay per PE, frozen during stall),
 *   just like weight and lock. This ensures each PE sees the correct propagate value
 *   matching the data that arrives at the same time.
 *   Used for BOTH compute weight selection AND capture buffer selection.
 *
 * Compute:
 *   out_psum = in_ifmap * weight_from_active_buffer + in_psum
 *   ifmap and psum are combinational (pipeline registers are in TwistMesh between PEs).
 */
class TwistPE[T <: Data](inputType: T, weightType: T, outputType: T, accType: T)
                         (implicit ev: Arithmetic[T]) extends Module {
  import ev._

  val io = IO(new Bundle {
    // Weight load path
    val in_weight  = Input(weightType)
    val out_weight = Output(weightType)
    val in_lock    = Input(Bool())
    val out_lock   = Output(Bool())

    // Compute data path
    val in_ifmap   = Input(inputType)
    val out_ifmap  = Output(inputType)
    val in_psum    = Input(outputType)
    val out_psum   = Output(outputType)

    // Control — pipelined left-to-right (NOT broadcast)
    val in_propagate  = Input(UInt(1.W))
    val out_propagate = Output(UInt(1.W))
    val in_valid      = Input(Bool())

    // Debug
    val debug_buffer1 = Output(inputType)
    val debug_buffer2 = Output(inputType)
    val debug_prop_delayed = Output(UInt(1.W))
    val debug_weight_sel = Output(inputType)
  })

  // Three registers — buffer width matches standard PE (cType = inputType for WS)
  val buffer1  = Reg(inputType)
  val buffer2  = Reg(inputType)
  val pass_reg = Reg(weightType)

  // Lock delay register: frozen during stall (in_valid=false)
  val lock_delayed = RegEnable(io.in_lock, false.B, io.in_valid)
  val lock_rise = io.in_lock && !lock_delayed

  // Propagate pipeline: 1-cycle delay, frozen during stall (same as lock/weight)
  val prop_delayed = RegEnable(io.in_propagate, io.in_valid)

  // Weight pass-through: update pass_reg when valid
  when(io.in_valid) {
    pass_reg := io.in_weight
  }

  // Delayed lock capture: capture pass_reg into inactive buffer
  // one cycle after lock rising edge.
  // Uses pipelined propagate (prop_delayed) for buffer selection —
  // this is the propagate value that arrived at the same time as the lock.
  val do_capture = RegNext(lock_rise && io.in_valid, false.B)
  val prop_at_capture = RegNext(io.in_propagate)  // 1-cycle delay, aligned with do_capture
  when(do_capture) {
    when(prop_at_capture === 0.U) {
      buffer2 := pass_reg  // prop=0: load inactive buffer2
    }.otherwise {
      buffer1 := pass_reg  // prop=1: load inactive buffer1
    }
  }

  // MAC computation: out_psum = in_psum + in_ifmap * weight
  // propagate=0: compute with buffer1 (active), load buffer2 (inactive)
  // propagate=1: compute with buffer2 (active), load buffer1 (inactive)
  // Uses in_propagate directly (combinational, same as standard PE).
  val weight_sel = Mux(io.in_propagate === 0.U, buffer1, buffer2)
  val mac_unit = Module(new MacUnit(inputType, weightType, outputType, outputType))
  mac_unit.io.in_a := io.in_ifmap
  mac_unit.io.in_b := weight_sel
  mac_unit.io.in_c := io.in_psum

  when(io.in_valid) {
    io.out_psum := mac_unit.io.out_d
  }.otherwise {
    io.out_psum := io.in_psum
  }

  // Pass-through outputs (weight/lock/propagate registered inside PE;
  // ifmap/psum are combinational — pipeline registers are in TwistMesh between PEs)
  io.out_weight    := pass_reg
  io.out_ifmap     := io.in_ifmap         // Combinational
  io.out_lock      := lock_delayed
  io.out_propagate := prop_delayed

  // Debug
  io.debug_buffer1 := buffer1
  io.debug_buffer2 := buffer2
  io.debug_prop_delayed := prop_delayed
  io.debug_weight_sel := weight_sel
}
