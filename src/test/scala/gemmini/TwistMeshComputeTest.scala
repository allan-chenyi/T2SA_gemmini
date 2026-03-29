package gemmini

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/**
 * Per-cycle compute trace for TwistMesh (8x8).
 *
 * Uses dualOp=true to test both LP1 and LP2.
 * Only CP1 (ifmap Red, psum Black) is used in production.
 */
class TwistMeshComputeSpec extends AnyFlatSpec with ChiselScalatestTester {
  val n = 8

  val bMatrix: Array[Array[Int]] = Array.tabulate(n, n)((i, j) => (i + 1) * 10 + (j + 1))
  val aMatrix: Array[Array[Int]] = Array.tabulate(n, n)((i, j) => i * n + j + 1)

  def peekS32(s: SInt): Long = { val v = s.peek().litValue; if (v.testBit(31)) (v - (BigInt(1) << 32)).toLong else v.toLong }

  def loadWeights(dut: TwistMesh[SInt], lp: Int, feedOrder: String): Unit = {
    val feed: Seq[Seq[Int]] = feedOrder match {
      case "col-major" => (0 until n).map { k => (0 until n).map { r => bMatrix(r)(k) } }
      case "row-major" => (0 until n).map { k => (0 until n).map { r => bMatrix(k)(r) } }
    }
    dut.reset.poke(true.B); dut.clock.step(1)
    dut.reset.poke(false.B); dut.clock.step(1)
    dut.io.in_valid.poke(true.B); dut.io.in_propagate.poke(1.U)  // propagate=1 → load into buffer1
    dut.io.load_pattern.get.poke((lp - 1).U)
    for (r <- 0 until n) { dut.io.in_ifmap(r).poke(0.S); dut.io.in_psum(r).poke(0.S) }
    for (k <- 0 until 2 * n) {
      for (r <- 0 until n) dut.io.in_weight(r).poke(feed(k % n)(r).S)
      for (r <- 0 until n) dut.io.in_lock(r).poke((k < n && r == k).B)
      dut.clock.step(1)
    }
    for (_ <- 0 until 4) {
      for (r <- 0 until n) { dut.io.in_weight(r).poke(0.S); dut.io.in_lock(r).poke(false.B) }
      dut.clock.step(1)
    }
  }

  def runTrace(lp: Int, feedOrder: String): Unit = {
    val lpName = if (lp == 1) "weight BLACK, lock RED" else "weight RED, lock BLACK"
    println(s"\n================================================================")
    println(s"LP$lp ($lpName) + CP1 (ifmap RED, psum BLACK) + $feedOrder B")
    println(s"================================================================")

    test(new TwistMesh(n, dualOp = true, SInt(8.W), SInt(8.W), SInt(32.W), SInt(32.W))) { dut =>
      loadWeights(dut, lp, feedOrder)

      // Print loaded buffer1
      println("  buffer1:")
      print("        ")
      for (c <- 0 until n) print(f"  col$c%d")
      println()
      for (r <- 0 until n) {
        print(f"  row$r: ")
        for (c <- 0 until n) print(f"${peekS32(dut.io.debug_buffer1(r)(c))}%6d")
        println()
      }

      // Compute: feed A row-major
      dut.io.in_valid.poke(true.B); dut.io.in_propagate.poke(0.U)
      for (r <- 0 until n) { dut.io.in_lock(r).poke(false.B); dut.io.in_weight(r).poke(0.S) }

      println(s"\n  Compute trace (A row-major, cycle k → output row of C):")
      for (k <- 0 until n) {
        for (r <- 0 until n) {
          dut.io.in_ifmap(r).poke(aMatrix(k)(r).S)
          dut.io.in_psum(r).poke(0.S)
        }
        val output = (0 until n).map(r => peekS32(dut.io.out_psum(r)))
        println(s"    Cycle $k: A[$k]=[${(0 until n).map(r => aMatrix(k)(r)).mkString(",")}] → out=[${output.mkString(", ")}]")
        dut.clock.step(1)
      }

      println()
    }
  }

  behavior of "TwistMesh Compute Trace"

  // Production-relevant traces
  it should "trace LP1 + row-major B (production: A*B)" in { runTrace(1, "row-major") }
  it should "trace LP2 + col-major B (production: A*B via compensation)" in { runTrace(2, "col-major") }

  // Cross-check traces
  it should "trace LP1 + col-major B (produces A*BT)" in { runTrace(1, "col-major") }
  it should "trace LP2 + row-major B (produces A*BT)" in { runTrace(2, "row-major") }
}
