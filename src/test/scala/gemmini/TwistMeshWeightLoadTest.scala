package gemmini

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class TwistMeshWeightLoadSpec extends AnyFlatSpec with ChiselScalatestTester {
  val n = 8
  val bMatrix: Seq[Seq[Int]] = (0 until n).map { i =>
    (0 until n).map { j => (i + 1) * 10 + (j + 1) }
  }

  def peekSigned32(signal: SInt): Long = {
    val v = signal.peek().litValue
    if (v.testBit(31)) (v - (BigInt(1) << 32)).toLong else v.toLong
  }

  def runWeightLoadTest(dut: TwistMesh[SInt], lp: Int, feedOrder: String): Unit = {
    val feedSequence: Seq[Seq[Int]] = feedOrder match {
      case "col-major" =>
        (0 until n).map { k => (0 until n).map { r => bMatrix(r)(k) } }
      case "row-major" =>
        (0 until n).map { k => (0 until n).map { r => bMatrix(k)(r) } }
    }

    println(s"\n========================================")
    println(s"LP$lp + $feedOrder feed")
    println(s"========================================")

    dut.reset.poke(true.B); dut.clock.step(1)
    dut.reset.poke(false.B); dut.clock.step(1)
    dut.io.in_valid.poke(true.B)
    dut.io.in_propagate.poke(1.U)  // propagate=1 → load into buffer1
    // Set load pattern for dual-op mesh
    dut.io.load_pattern.get.poke((lp - 1).U)  // LP1→0, LP2→1
    for (ci <- 0 until n) { dut.io.in_ifmap(ci).poke(0.S); dut.io.in_psum(ci).poke(0.S) }

    for (k <- 0 until 2 * n) {
      val fi = k % n
      for (r <- 0 until n) dut.io.in_weight(r).poke(feedSequence(fi)(r).S)
      for (r <- 0 until n) dut.io.in_lock(r).poke((k < n && r == k).B)
      dut.clock.step(1)
    }
    for (_ <- 0 until 4) {
      for (r <- 0 until n) { dut.io.in_weight(r).poke(0.S); dut.io.in_lock(r).poke(false.B) }
      dut.clock.step(1)
    }

    println(s"\nbuffer1:")
    print("        ")
    for (c <- 0 until n) print(f"  col$c%d")
    println()
    for (r <- 0 until n) {
      print(f"  row$r: ")
      for (c <- 0 until n) print(f"${peekSigned32(dut.io.debug_buffer1(r)(c))}%6d")
      println()
    }
    println()
  }

  behavior of "TwistMesh Weight Load"

  // Use dualOp=true so we can test both LP1 and LP2 via load_pattern
  for (lp <- Seq(1, 2); feed <- Seq("col-major", "row-major")) {
    it should s"load LP$lp $feed" in {
      test(new TwistMesh(n, dualOp = true, SInt(8.W), SInt(8.W), SInt(32.W), SInt(32.W)))
        .withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        runWeightLoadTest(dut, lp, feed)
      }
    }
  }
}
