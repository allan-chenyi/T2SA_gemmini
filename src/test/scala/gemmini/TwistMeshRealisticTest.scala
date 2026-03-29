package gemmini

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/**
 * Realistic test: mimics TwistMeshWithDelays behavior exactly.
 *
 * In the real system:
 *   1. PRELOAD: n cycles of weight+lock with in_valid=true
 *   2. PAUSE: in_valid=false while waiting for COMPUTE req (variable length)
 *   3. COMPUTE: n cycles of ifmap with in_valid=true
 *
 * NO explicit propagation cycles after PRELOAD.
 * Lock/weight freeze during pause (RegEnable gated by in_valid).
 * When COMPUTE starts, frozen signals resume propagating.
 *
 * Key question: does the weight input during COMPUTE corrupt pass_reg
 * for columns that haven't received their lock yet?
 */
class TwistMeshRealisticSpec extends AnyFlatSpec with ChiselScalatestTester {
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

  behavior of "TwistMesh Realistic"

  it should "trace PRELOAD→PAUSE→COMPUTE like real MeshWithDelays" in {
    test(new TwistMesh(n, dualOp = false, SInt(8.W), SInt(8.W), SInt(20.W), SInt(32.W),
                       max_simultaneous_matmuls = 4)) { dut =>

      dut.reset.poke(true.B); dut.clock.step(1)
      dut.reset.poke(false.B); dut.clock.step(1)

      // === PRELOAD: n cycles weight+lock, prop=0 ===
      println("\n=== PRELOAD W1 (prop=0, n=4 cycles) ===")
      dut.io.in_valid.poke(true.B)
      dut.io.in_propagate.poke(0.U)
      dut.io.in_last.poke(false.B)
      dut.io.in_id.poke(0.U)
      for (r <- 0 until n) { dut.io.in_ifmap(r).poke(0.S); dut.io.in_psum(r).poke(0.S) }

      for (k <- 0 until n) {
        for (r <- 0 until n) {
          dut.io.in_weight(r).poke(W1(k)(r).S)
          dut.io.in_lock(r).poke((r == k).B)
        }
        println(s"\n--- PRELOAD cycle $k ---")
        println(s"  in_weight=[${(0 until n).map(r => peekS(dut.io.in_weight(r))).mkString(",")}]")
        println(s"  in_lock=[${(0 until n).map(r => dut.io.in_lock(r).peek().litToBoolean).mkString(",")}]")
        printPEGrid(dut, "buffer1", (r, c) => peekS(dut.io.debug_buffer1(r)(c)))
        dut.clock.step(1)
      }

      println("\n=== After PRELOAD (before pause) ===")
      printPEGrid(dut, "buffer1", (r, c) => peekS(dut.io.debug_buffer1(r)(c)))
      printPEGrid(dut, "buffer2", (r, c) => peekS(dut.io.debug_buffer2(r)(c)))

      // === PAUSE: in_valid=false for a few cycles (like real system) ===
      println("\n=== PAUSE (in_valid=false, 3 cycles) ===")
      dut.io.in_valid.poke(false.B)
      for (r <- 0 until n) { dut.io.in_weight(r).poke(0.S); dut.io.in_lock(r).poke(false.B) }
      for (p <- 0 until 3) {
        println(s"\n--- PAUSE cycle $p ---")
        printPEGrid(dut, "buffer1", (r, c) => peekS(dut.io.debug_buffer1(r)(c)))
        dut.clock.step(1)
      }

      println("\n=== After PAUSE ===")
      printPEGrid(dut, "buffer1", (r, c) => peekS(dut.io.debug_buffer1(r)(c)))

      // === COMPUTE: n cycles ifmap, prop=1 (flipped from PRELOAD's prop=0) ===
      // prop=0 loaded buffer2, so prop=1 computes with buffer2
      println("\n=== COMPUTE A1*W1 (prop=1, weight=0 during compute) ===")
      dut.io.in_valid.poke(true.B)
      dut.io.in_propagate.poke(1.U)

      val totalCycles = 2 * n + 2
      for (k <- 0 until totalCycles) {
        val isFeeding = k < n
        val isLast = k == n - 1
        dut.io.in_last.poke(isLast.B)
        for (r <- 0 until n) {
          dut.io.in_ifmap(r).poke((if (isFeeding) A1(k)(r) else 0).S)
          dut.io.in_psum(r).poke(0.S)
          dut.io.in_weight(r).poke(0.S)  // no weight during compute
          dut.io.in_lock(r).poke(false.B)
        }

        val phase = if (isFeeding) s"feed A1[$k]" else s"drain ${k - n}"
        println(s"\n--- COMPUTE cycle $k ($phase) ---")
        printPEGrid(dut, "buffer1", (r, c) => peekS(dut.io.debug_buffer1(r)(c)))
        printPEGrid(dut, "PE weight_sel", (r, c) => peekS(dut.io.debug_pe_weight_sel(r)(c)))

        val out = (0 until n).map(r => peekS(dut.io.out_psum(r)))
        val valid = dut.io.out_valid.peek().litToBoolean
        val last = dut.io.out_last.peek().litToBoolean
        println(s"  out_psum=[${out.mkString(", ")}] valid=$valid last=$last")

        dut.clock.step(1)
      }

      println("\n\n=== COMPARISON: Now with propagation cycles ===")
      // Reset and redo with propagation cycles
      dut.reset.poke(true.B); dut.clock.step(1)
      dut.reset.poke(false.B); dut.clock.step(1)

      // PRELOAD with propagation
      dut.io.in_valid.poke(true.B)
      dut.io.in_propagate.poke(0.U)
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
      // Extra propagation cycles
      for (_ <- 0 until n) {
        for (r <- 0 until n) { dut.io.in_weight(r).poke(0.S); dut.io.in_lock(r).poke(false.B) }
        dut.clock.step(1)
      }

      println("After PRELOAD + propagation:")
      printPEGrid(dut, "buffer1 (with prop cycles)", (r, c) => peekS(dut.io.debug_buffer1(r)(c)))

      // Pause
      dut.io.in_valid.poke(false.B)
      dut.clock.step(n)

      // COMPUTE (prop=1, flipped from PRELOAD's prop=0)
      dut.io.in_valid.poke(true.B)
      dut.io.in_propagate.poke(1.U)
      for (k <- 0 until totalCycles) {
        val isFeeding = k < n
        val isLast = k == n - 1
        dut.io.in_last.poke(isLast.B)
        for (r <- 0 until n) {
          dut.io.in_ifmap(r).poke((if (isFeeding) A1(k)(r) else 0).S)
          dut.io.in_psum(r).poke(0.S)
          dut.io.in_weight(r).poke(0.S)
          dut.io.in_lock(r).poke(false.B)
        }

        val out = (0 until n).map(r => peekS(dut.io.out_psum(r)))
        val valid = dut.io.out_valid.peek().litToBoolean
        val last = dut.io.out_last.peek().litToBoolean
        if (valid) {
          println(s"  [with prop] cycle $k: out=[${out.mkString(", ")}] last=$last")
        }
        dut.clock.step(1)
      }

      println("\n  Expected C1 = W1 (A1=identity):")
      for (r <- 0 until n) {
        print(s"    row$r: ")
        for (c <- 0 until n) print(f"${W1(r)(c)}%6d ")
        println()
      }
    }
  }
}
