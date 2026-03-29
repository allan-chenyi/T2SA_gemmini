package gemmini

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/**
 * Verification: TwistMesh computes A*B correctly with different LP + feed combos.
 *
 * Production usage (row-major B from scratchpad):
 *   LP1 + row-major B → A*B    (single-op and dual-op default)
 *   LP2 + row-major B → A*B^T  (but LP2 + col-major B → A*B, used for B col-major layout)
 *
 * We test 4 relevant configurations (all CP1, row-major A):
 *   1. LP1 + row-major B → A*B exact         (production: single-op / dual-op default)
 *   2. LP2 + col-major B → A*B exact         (production: dual-op B col-major compensation)
 *   3. LP1 + col-major B → A*B^T exact       (verify: wrong result if B col-major on LP1)
 *   4. LP2 + row-major B → A*B^T exact       (verify: wrong result if B row-major on LP2)
 */
class TwistMeshABVerifySpec extends AnyFlatSpec with ChiselScalatestTester {
  val n = 8

  val bMatrix: Array[Array[Int]] = Array.tabulate(n, n)((i, j) => (i + 1) * 10 + (j + 1))
  val aMatrix: Array[Array[Int]] = Array.tabulate(n, n)((i, j) => i * n + j + 1)

  def matmul(a: Array[Array[Int]], b: Array[Array[Int]]): Array[Array[Long]] =
    Array.tabulate(a.length, b(0).length)((i, j) =>
      (0 until a(0).length).map(k => a(i)(k).toLong * b(k)(j).toLong).sum)

  val bT: Array[Array[Int]] = Array.tabulate(n, n)((i, j) => bMatrix(j)(i))
  val AB:  Array[Array[Long]] = matmul(aMatrix, bMatrix)
  val ABT: Array[Array[Long]] = matmul(aMatrix, bT)

  def peekS32(s: SInt): Long = {
    val v = s.peek().litValue
    if (v.testBit(31)) (v - (BigInt(1) << 32)).toLong else v.toLong
  }

  def loadWeights(dut: TwistMesh[SInt], lp: Int, feedOrder: String): Unit = {
    val feed: Seq[Seq[Int]] = feedOrder match {
      case "col-major" => (0 until n).map { k => (0 until n).map { r => bMatrix(r)(k) } }
      case "row-major" => (0 until n).map { k => (0 until n).map { r => bMatrix(k)(r) } }
    }
    dut.reset.poke(true.B); dut.clock.step(1)
    dut.reset.poke(false.B); dut.clock.step(1)
    dut.io.in_valid.poke(true.B); dut.io.in_propagate.poke(1.U)  // propagate=1 → load into buffer1
    dut.io.load_pattern.get.poke((lp - 1).U)  // LP1→0, LP2→1
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

  def runVerify(lp: Int, feedOrder: String,
                expectedName: String, expected: Array[Array[Long]]): Unit = {
    println(s"\n========================================")
    println(s"LP$lp + $feedOrder B → $expectedName")
    println(s"========================================")

    test(new TwistMesh(n, dualOp = true, SInt(8.W), SInt(8.W), SInt(32.W), SInt(32.W))) { dut =>
      loadWeights(dut, lp, feedOrder)

      // Print buffer1 row 0
      println("  buffer1 row0: " + (0 until n).map(c => peekS32(dut.io.debug_buffer1(0)(c))).mkString(", "))

      dut.io.in_valid.poke(true.B); dut.io.in_propagate.poke(0.U)
      for (r <- 0 until n) { dut.io.in_lock(r).poke(false.B); dut.io.in_weight(r).poke(0.S) }

      var allPass = true
      for (k <- 0 until n) {
        for (r <- 0 until n) {
          dut.io.in_ifmap(r).poke(aMatrix(k)(r).S)
          dut.io.in_psum(r).poke(0.S)
        }

        val output = (0 until n).map(r => peekS32(dut.io.out_psum(r)))
        val pass = (0 until n).forall(r => output(r) == expected(k)(r))

        val status = if (pass) "PASS" else "FAIL"
        println(s"  Cycle $k: output=[${output.mkString(", ")}]  $status")

        if (!pass) {
          println(s"           expect=[${expected(k).mkString(", ")}]")
          allPass = false
        }

        dut.clock.step(1)
      }

      println(s"\n  Reference ($expectedName)[0] = [${expected(0).mkString(", ")}]")

      if (allPass) println(s"  >>> ALL PASS <<<")
      else         println(s"  >>> FAILED <<<")
      assert(allPass, s"LP$lp + $feedOrder failed to produce $expectedName")
    }
  }

  behavior of "TwistMesh A*B Verification"

  // Production cases: these must produce A*B
  it should "LP1 + row-major B = A*B (single-op / dual-op default)" in {
    runVerify(1, "row-major", "A*B", AB)
  }
  it should "LP2 + col-major B = A*B (dual-op B col-major compensation)" in {
    runVerify(2, "col-major", "A*B", AB)
  }

  // Verification cases: these produce A*B^T (wrong if target is A*B)
  it should "LP1 + col-major B = A*BT (expected: NOT A*B)" in {
    runVerify(1, "col-major", "A*B^T", ABT)
  }
  it should "LP2 + row-major B = A*BT (expected: NOT A*B)" in {
    runVerify(2, "row-major", "A*B^T", ABT)
  }
}
