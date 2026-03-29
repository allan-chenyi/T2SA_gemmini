package gemmini

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/**
 * Back-to-back matmul test on 4x4 TwistMesh.
 * Uses true double-buffer convention (matches standard Gemmini PE):
 *   prop=0: load buffer2 (inactive), compute with buffer1 (active)
 *   prop=1: load buffer1 (inactive), compute with buffer2 (active)
 *
 * So the sequence is:
 *   Matmul 1: PRELOAD W1 with prop=0 (load buffer2), COMPUTE A1*W1 with prop=1 (use buffer2)
 *   Matmul 2: PRELOAD W2 with prop=1 (load buffer1), COMPUTE A2*W2 with prop=0 (use buffer1)
 */
class TwistMeshBackToBackSpec extends AnyFlatSpec with ChiselScalatestTester {
  val n = 4

  // Simple test matrices
  val W1 = Array.tabulate(n, n)((i, j) => (i + 1) * 10 + (j + 1))  // 11..44
  val W2 = Array.tabulate(n, n)((i, j) => (i + 1) + (j + 1) * 10)  // transposed pattern
  val A1 = Array.tabulate(n, n)((i, j) => if (i == j) 1 else 0)     // identity
  val A2 = Array.tabulate(n, n)((i, j) => if (i == j) 2 else 0)     // 2*identity

  def peekS(s: SInt): Long = {
    val v = s.peek().litValue
    val w = s.getWidth
    if (v.testBit(w - 1)) (v - (BigInt(1) << w)).toLong else v.toLong
  }
  def peekU(u: UInt): Long = u.peek().litValue.toLong

  def printBuffers(dut: TwistMesh[SInt], label: String): Unit = {
    println(s"  $label buffer1:")
    for (r <- 0 until n) {
      print(s"    row$r: ")
      for (c <- 0 until n) print(f"${peekS(dut.io.debug_buffer1(r)(c))}%6d ")
      println()
    }
    println(s"  $label buffer2:")
    for (r <- 0 until n) {
      print(s"    row$r: ")
      for (c <- 0 until n) print(f"${peekS(dut.io.debug_buffer2(r)(c))}%6d ")
      println()
    }
  }

  /** Load weight matrix into mesh with given propagate value */
  def loadWeight(dut: TwistMesh[SInt], W: Array[Array[Int]], prop: Int): Unit = {
    dut.io.in_valid.poke(true.B)
    dut.io.in_propagate.poke(prop.U)
    dut.io.in_last.poke(false.B)
    for (r <- 0 until n) { dut.io.in_ifmap(r).poke(0.S); dut.io.in_psum(r).poke(0.S) }

    // Feed weight rows with staggered lock
    for (k <- 0 until n) {
      for (r <- 0 until n) {
        dut.io.in_weight(r).poke(W(k)(r).S)
        dut.io.in_lock(r).poke((r == k).B)
      }
      dut.clock.step(1)
    }
    // Propagation cycles (lock/weight need n-1 more cycles to reach all PEs)
    for (_ <- 0 until n) {
      for (r <- 0 until n) { dut.io.in_weight(r).poke(0.S); dut.io.in_lock(r).poke(false.B) }
      dut.clock.step(1)
    }

    // Pause (like real system between PRELOAD and COMPUTE) — flush valid pipeline
    dut.io.in_valid.poke(false.B)
    dut.clock.step(n)
  }

  /** Compute A*W with given propagate value, collect n output rows */
  def compute(dut: TwistMesh[SInt], A: Array[Array[Int]], prop: Int): Array[Array[Long]] = {
    dut.io.in_valid.poke(true.B)
    dut.io.in_propagate.poke(prop.U)
    val result = Array.ofDim[Long](n, n)
    var resultIdx = 0

    val totalCycles = 2 * n + 2
    for (k <- 0 until totalCycles) {
      val isFeeding = k < n
      val isLast = k == n - 1
      dut.io.in_last.poke(isLast.B)
      for (r <- 0 until n) {
        dut.io.in_ifmap(r).poke((if (isFeeding) A(k)(r) else 0).S)
        dut.io.in_psum(r).poke(0.S)
        dut.io.in_weight(r).poke(0.S)
        dut.io.in_lock(r).poke(false.B)
      }

      // Check output before clock edge
      val valid = dut.io.out_valid.peek().litToBoolean
      if (valid && resultIdx < n) {
        for (r <- 0 until n) result(resultIdx)(r) = peekS(dut.io.out_psum(r))
        resultIdx += 1
      }

      dut.clock.step(1)
    }
    // Check one more cycle after last step
    val valid = dut.io.out_valid.peek().litToBoolean
    if (valid && resultIdx < n) {
      for (r <- 0 until n) result(resultIdx)(r) = peekS(dut.io.out_psum(r))
      resultIdx += 1
    }

    assert(resultIdx == n, s"Expected $n output rows but got $resultIdx")
    result
  }

  def printMatrix(label: String, m: Array[Array[Long]]): Unit = {
    println(s"  $label:")
    for (r <- 0 until n) {
      print(s"    row$r: ")
      for (c <- 0 until n) print(f"${m(r)(c)}%6d ")
      println()
    }
  }
  def printMatrix(label: String, m: Array[Array[Int]]): Unit = {
    println(s"  $label:")
    for (r <- 0 until n) {
      print(s"    row$r: ")
      for (c <- 0 until n) print(f"${m(r)(c)}%6d ")
      println()
    }
  }

  /** Software matmul for reference */
  def matmul(A: Array[Array[Int]], B: Array[Array[Int]]): Array[Array[Int]] = {
    Array.tabulate(n, n)((i, j) => (0 until n).map(k => A(i)(k) * B(k)(j)).sum)
  }

  behavior of "TwistMesh Back-to-Back"

  it should "do two back-to-back matmuls with standard propagate convention" in {
    test(new TwistMesh(n, dualOp = false, SInt(8.W), SInt(8.W), SInt(20.W), SInt(32.W),
                       max_simultaneous_matmuls = 4)) { dut =>

      dut.reset.poke(true.B); dut.clock.step(1)
      dut.reset.poke(false.B); dut.clock.step(1)

      // === Matmul 1: PRELOAD prop=0 (loads buffer2), COMPUTE prop=1 (reads buffer2) ===
      println("\n=== Matmul 1: PRELOAD W1 (prop=0 → buffer2), COMPUTE A1*W1 (prop=1 → use buffer2) ===")
      loadWeight(dut, W1, prop = 0)
      printBuffers(dut, "After load W1")

      val C1 = compute(dut, A1, prop = 1)  // prop flips for COMPUTE
      val expected1 = matmul(A1, W1)
      printMatrix("C1 actual (A1*W1)", C1)
      printMatrix("C1 expected", expected1)

      // Verify
      var pass1 = true
      for (r <- 0 until n; c <- 0 until n) {
        if (C1(r)(c) != expected1(r)(c)) { pass1 = false }
      }
      println(s"  Matmul 1: ${if (pass1) "PASS" else "FAIL"}")

      // === Matmul 2: PRELOAD prop=1 (loads buffer1), COMPUTE prop=0 (reads buffer1) ===
      println("\n=== Matmul 2: PRELOAD W2 (prop=1 → buffer1), COMPUTE A2*W2 (prop=0 → use buffer1) ===")
      loadWeight(dut, W2, prop = 1)
      printBuffers(dut, "After load W2")

      val C2 = compute(dut, A2, prop = 0)  // prop flips for COMPUTE
      val expected2 = matmul(A2, W2)
      printMatrix("C2 actual (A2*W2)", C2)
      printMatrix("C2 expected", expected2)

      var pass2 = true
      for (r <- 0 until n; c <- 0 until n) {
        if (C2(r)(c) != expected2(r)(c)) { pass2 = false }
      }
      println(s"  Matmul 2: ${if (pass2) "PASS" else "FAIL"}")

      assert(pass1, "Matmul 1 failed")
      assert(pass2, "Matmul 2 failed")
      println("\n=== ALL PASS ===")
    }
  }
}
