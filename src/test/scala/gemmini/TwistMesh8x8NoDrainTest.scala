package gemmini

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/**
 * 8x8 test WITHOUT drain cycles — matches real TwistMeshWithDelays behavior.
 * n fire cycles for PRELOAD, then immediately pause (in_valid=false).
 * Tests whether col_valid's natural delay provides enough propagation time.
 */
class TwistMesh8x8NoDrainSpec extends AnyFlatSpec with ChiselScalatestTester {
  val n = 8

  val W1 = Array.tabulate(n, n)((i, j) => (i * n + j + 1) % 17 - 8)
  val A1 = Array.tabulate(n, n)((i, j) => if (i == j) 1 else 0)  // identity

  def peekS(s: SInt): Long = s.peek().litValue.toLong

  def matmul(A: Array[Array[Int]], B: Array[Array[Int]]): Array[Array[Int]] =
    Array.tabulate(n, n)((i, j) => (0 until n).map(k => A(i)(k) * B(k)(j)).sum)

  behavior of "TwistMesh 8x8 no drain"

  it should "work with only n fire cycles (no drain)" in {
    test(new TwistMesh(n, dualOp = false, SInt(8.W), SInt(8.W), SInt(20.W), SInt(32.W),
                       max_simultaneous_matmuls = 4)) { dut =>

      dut.reset.poke(true.B); dut.clock.step(1)
      dut.reset.poke(false.B); dut.clock.step(1)

      // === PRELOAD: exactly n fire cycles, NO drain ===
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

      // IMMEDIATELY pause (no drain cycles!) — like real TwistMeshWithDelays
      dut.io.in_valid.poke(false.B)
      for (r <- 0 until n) { dut.io.in_weight(r).poke(0.S); dut.io.in_lock(r).poke(false.B) }
      dut.clock.step(n)  // wait for pipeline to settle

      // Print buffer2 (prop=0 loads buffer2)
      println("buffer2 after PRELOAD (no drain):")
      for (r <- 0 until n) {
        print(s"  row$r: ")
        for (c <- 0 until n) print(f"${peekS(dut.io.debug_buffer2(r)(c))}%5d ")
        println()
      }

      // Expected: PE(r,c) captures W[(r+c)%n][r]
      println("\nExpected buffer2 (W[(r+c)%n][r]):")
      for (r <- 0 until n) {
        print(s"  row$r: ")
        for (c <- 0 until n) print(f"${W1((r + c) % n)(r)}%5d ")
        println()
      }

      // Check buffer2
      var bufOk = true
      for (r <- 0 until n; c <- 0 until n) {
        val actual = peekS(dut.io.debug_buffer2(r)(c))
        val expected = W1((r + c) % n)(r)
        if (actual != expected) {
          println(s"  BUFFER MISMATCH PE($r,$c): actual=$actual expected=$expected")
          bufOk = false
        }
      }
      println(s"Buffer check: ${if (bufOk) "ALL CORRECT" else "HAS ERRORS"}")

      // === COMPUTE: prop=1 reads buffer2 ===
      dut.io.in_valid.poke(true.B)
      dut.io.in_propagate.poke(1.U)
      val result = Array.ofDim[Long](n, n)
      var resultIdx = 0

      val totalCycles = 2 * n + 2
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
        val valid = dut.io.out_valid.peek().litToBoolean
        if (valid && resultIdx < n) {
          for (r <- 0 until n) result(resultIdx)(r) = peekS(dut.io.out_psum(r))
          resultIdx += 1
        }
        dut.clock.step(1)
      }
      // Extra cycle
      if (dut.io.out_valid.peek().litToBoolean && resultIdx < n) {
        for (r <- 0 until n) result(resultIdx)(r) = peekS(dut.io.out_psum(r))
        resultIdx += 1
      }

      val expected = matmul(A1, W1)
      println(s"\nCollected $resultIdx output rows")
      println("Result:")
      for (r <- 0 until n) {
        print(s"  row$r: ")
        for (c <- 0 until n) print(f"${result(r)(c)}%5d ")
        println()
      }
      println("Expected (A1*W1 = W1):")
      for (r <- 0 until n) {
        print(s"  row$r: ")
        for (c <- 0 until n) print(f"${expected(r)(c)}%5d ")
        println()
      }

      var pass = true
      for (r <- 0 until n; c <- 0 until n) {
        if (result(r)(c) != expected(r)(c)) {
          println(s"  COMPUTE MISMATCH ($r,$c): actual=${result(r)(c)} expected=${expected(r)(c)}")
          pass = false
        }
      }
      println(s"\nCompute: ${if (pass) "PASS" else "FAIL"}")
      assert(bufOk, "Buffer loading failed")
      assert(pass, "Compute failed")
    }
  }
}
