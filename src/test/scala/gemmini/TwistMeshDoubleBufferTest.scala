package gemmini

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/**
 * Detailed per-PE trace for a single matmul on 4x4 TwistMesh.
 * Load W1, then compute A1*W1 with A1=identity. Print every PE's state each cycle.
 */
class TwistMeshDoubleBufferSpec extends AnyFlatSpec with ChiselScalatestTester {
  val n = 4

  val W1 = Array.tabulate(n, n)((i, j) => (i + 1) * 10 + (j + 1))
  val A1 = Array.tabulate(n, n)((i, j) => if (i == j) 1 else 0)

  def peekS(s: SInt): Long = {
    val v = s.peek().litValue
    val w = s.getWidth
    if (v.testBit(w - 1)) (v - (BigInt(1) << w)).toLong else v.toLong
  }
  def peekU(u: UInt): Long = u.peek().litValue.toLong

  def printPEGrid(dut: TwistMesh[SInt], label: String, getter: (Int, Int) => Long): Unit = {
    println(s"  $label:")
    print("        ")
    for (c <- 0 until n) print(f"  col$c%d  ")
    println()
    for (r <- 0 until n) {
      print(f"  row$r: ")
      for (c <- 0 until n) print(f"${getter(r, c)}%6d ")
      println()
    }
  }

  behavior of "TwistMesh Double Buffer"

  it should "trace single matmul per-PE per-cycle" in {
    test(new TwistMesh(n, dualOp = false, SInt(8.W), SInt(8.W), SInt(20.W), SInt(32.W),
                       max_simultaneous_matmuls = 4)) { dut =>

      // Reset
      dut.reset.poke(true.B); dut.clock.step(1)
      dut.reset.poke(false.B); dut.clock.step(1)

      // === Load W1 with propagate=1 (load into buffer1) ===
      println("\n=== Load W1 (propagate=1 → load into buffer1) ===")
      dut.io.in_valid.poke(true.B)
      dut.io.in_propagate.poke(1.U)
      dut.io.in_last.poke(false.B)
      dut.io.in_id.poke(0.U)
      for (r <- 0 until n) { dut.io.in_ifmap(r).poke(0.S); dut.io.in_psum(r).poke(0.S) }

      for (k <- 0 until n) {
        for (r <- 0 until n) {
          dut.io.in_weight(r).poke(W1(k)(r).S)
          dut.io.in_lock(r).poke((r == k).B)
        }
        dut.clock.step(1)
      }
      // Propagation cycles
      for (_ <- 0 until n) {
        for (r <- 0 until n) { dut.io.in_weight(r).poke(0.S); dut.io.in_lock(r).poke(false.B) }
        dut.clock.step(1)
      }

      println("  buffer1 after load:")
      printPEGrid(dut, "buffer1", (r, c) => peekS(dut.io.debug_buffer1(r)(c)))
      printPEGrid(dut, "buffer2", (r, c) => peekS(dut.io.debug_buffer2(r)(c)))

      // === Compute A1*W1 with propagate=0 (use buffer1) ===
      println("\n=== Compute A1*W1 (propagate=0 → use buffer1, A1=identity) ===")
      dut.io.in_propagate.poke(0.U)
      dut.io.in_id.poke(1.U)

      val totalCycles = 2 * n + 2  // enough to see all outputs
      for (k <- 0 until totalCycles) {
        // Feed ifmap for first n cycles, then 0
        val isFeeding = k < n
        val isLast = k == n - 1
        dut.io.in_last.poke(isLast.B)
        for (r <- 0 until n) {
          val ifmapVal = if (isFeeding) A1(k)(r) else 0
          dut.io.in_ifmap(r).poke(ifmapVal.S)
          dut.io.in_psum(r).poke(0.S)
          dut.io.in_weight(r).poke(0.S)
          dut.io.in_lock(r).poke(false.B)
        }

        // Print state BEFORE clock edge
        val phase = if (isFeeding) s"feed A1[$k]" else s"drain ${k - n}"
        println(s"\n--- Cycle $k ($phase) ---")
        println(s"  in_propagate=${peekU(dut.io.in_propagate)}, in_valid=${dut.io.in_valid.peek().litToBoolean}, in_last=${dut.io.in_last.peek().litToBoolean}")
        println(s"  in_ifmap=[${(0 until n).map(r => peekS(dut.io.in_ifmap(r))).mkString(",")}]")

        printPEGrid(dut, "PE in_ifmap", (r, c) => peekS(dut.io.debug_pe_in_ifmap(r)(c)))
        printPEGrid(dut, "PE weight_sel", (r, c) => peekS(dut.io.debug_pe_weight_sel(r)(c)))
        printPEGrid(dut, "PE prop_delayed", (r, c) => peekU(dut.io.debug_pe_prop_delayed(r)(c)))
        printPEGrid(dut, "PE out_psum", (r, c) => peekS(dut.io.debug_pe_out_psum(r)(c)))

        val out = (0 until n).map(r => peekS(dut.io.out_psum(r)))
        val valid = dut.io.out_valid.peek().litToBoolean
        val last = dut.io.out_last.peek().litToBoolean
        println(s"  mesh out_psum=[${out.mkString(", ")}] valid=$valid last=$last")

        dut.clock.step(1)
      }

      println("\n  Expected C1 = W1 (since A1=identity):")
      for (r <- 0 until n) {
        print(s"    row$r: ")
        for (c <- 0 until n) print(f"${W1(r)(c)}%6d")
        println()
      }
    }
  }
}
