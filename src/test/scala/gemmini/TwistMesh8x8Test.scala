package gemmini

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/**
 * 8x8 back-to-back matmul test with per-cycle trace.
 * Uses true double-buffer convention:
 *   Matmul 1: PRELOAD prop=0 (load buffer2), COMPUTE prop=1 (use buffer2)
 *   Matmul 2: PRELOAD prop=1 (load buffer1), COMPUTE prop=0 (use buffer1)
 */
class TwistMesh8x8Spec extends AnyFlatSpec with ChiselScalatestTester {
  val n = 8

  // Simple test matrices
  val W1 = Array.tabulate(n, n)((i, j) => (i * n + j + 1) % 17 - 8)  // values in [-8, 8]
  val W2 = Array.tabulate(n, n)((i, j) => ((i + j) * 3 + 1) % 13 - 6)
  val A1 = Array.tabulate(n, n)((i, j) => if (i == j) 1 else 0)  // identity
  val A2 = Array.tabulate(n, n)((i, j) => if (i == j) 2 else 0)  // 2*identity

  def peekS(s: SInt): Long = s.peek().litValue.toLong

  def matmul(A: Array[Array[Int]], B: Array[Array[Int]]): Array[Array[Int]] = {
    Array.tabulate(n, n)((i, j) => (0 until n).map(k => A(i)(k) * B(k)(j)).sum)
  }

  def printMatrix(label: String, m: Array[Array[Int]]): Unit = {
    println(s"  $label:")
    for (r <- 0 until n) {
      print(s"    row$r: ")
      for (c <- 0 until n) print(f"${m(r)(c)}%5d ")
      println()
    }
  }
  def printMatrixL(label: String, m: Array[Array[Long]]): Unit = {
    println(s"  $label:")
    for (r <- 0 until n) {
      print(s"    row$r: ")
      for (c <- 0 until n) print(f"${m(r)(c)}%5d ")
      println()
    }
  }

  def printBuffer(dut: TwistMesh[SInt], bufNum: Int, label: String): Unit = {
    println(s"  $label buffer$bufNum:")
    for (r <- 0 until n) {
      print(s"    row$r: ")
      for (c <- 0 until n) {
        val v = if (bufNum == 1) peekS(dut.io.debug_buffer1(r)(c))
                else peekS(dut.io.debug_buffer2(r)(c))
        print(f"$v%5d ")
      }
      println()
    }
  }

  /** Load weight with per-cycle trace */
  def loadWeight(dut: TwistMesh[SInt], W: Array[Array[Int]], prop: Int, label: String): Unit = {
    println(s"\n--- loadWeight $label (prop=$prop) ---")
    dut.io.in_valid.poke(true.B)
    dut.io.in_propagate.poke(prop.U)
    dut.io.in_last.poke(false.B)
    dut.io.in_id.poke(0.U)
    for (r <- 0 until n) { dut.io.in_ifmap(r).poke(0.S); dut.io.in_psum(r).poke(0.S) }

    // Feed weight rows with staggered lock
    for (k <- 0 until n) {
      for (r <- 0 until n) {
        dut.io.in_weight(r).poke(W(k)(r).S)
        dut.io.in_lock(r).poke((r == k).B)
      }
      println(s"  load cycle $k: lock_row=$k, weight_row=${(0 until n).map(r => W(k)(r)).mkString(",")}")
      dut.clock.step(1)
    }
    // Propagation cycles
    for (k <- 0 until n) {
      for (r <- 0 until n) { dut.io.in_weight(r).poke(0.S); dut.io.in_lock(r).poke(false.B) }
      dut.clock.step(1)
    }

    // Pause
    dut.io.in_valid.poke(false.B)
    dut.clock.step(n)

    // Print loaded buffers
    val targetBuf = if (prop == 0) 2 else 1
    printBuffer(dut, targetBuf, s"After $label →")
  }

  /** Compute with per-cycle trace, collect output */
  def compute(dut: TwistMesh[SInt], A: Array[Array[Int]], prop: Int, label: String): Array[Array[Long]] = {
    println(s"\n--- compute $label (prop=$prop) ---")
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

      val valid = dut.io.out_valid.peek().litToBoolean
      val last = dut.io.out_last.peek().litToBoolean
      val out = (0 until n).map(r => peekS(dut.io.out_psum(r)))

      if (valid) {
        println(s"  compute cycle $k: out=[${out.mkString(",")}] last=$last (row $resultIdx)")
        if (resultIdx < n) {
          for (r <- 0 until n) result(resultIdx)(r) = out(r)
          resultIdx += 1
        }
      }
      dut.clock.step(1)
    }
    // One more cycle
    val valid = dut.io.out_valid.peek().litToBoolean
    if (valid && resultIdx < n) {
      val out = (0 until n).map(r => peekS(dut.io.out_psum(r)))
      println(s"  compute extra: out=[${out.mkString(",")}] (row $resultIdx)")
      for (r <- 0 until n) result(resultIdx)(r) = out(r)
      resultIdx += 1
    }

    println(s"  collected $resultIdx output rows (expected $n)")
    result
  }

  behavior of "TwistMesh 8x8"

  it should "do two back-to-back matmuls on 8x8 mesh" in {
    test(new TwistMesh(n, dualOp = false, SInt(8.W), SInt(8.W), SInt(20.W), SInt(32.W),
                       max_simultaneous_matmuls = 4)) { dut =>

      dut.reset.poke(true.B); dut.clock.step(1)
      dut.reset.poke(false.B); dut.clock.step(1)

      println("W1:"); printMatrix("", W1)
      println("A1 = identity")
      println("Expected C1 = A1*W1 = W1")

      // === Matmul 1 ===
      loadWeight(dut, W1, prop = 0, "W1")
      val C1 = compute(dut, A1, prop = 1, "A1*W1")
      val expected1 = matmul(A1, W1)
      printMatrixL("C1 actual", C1)
      printMatrix("C1 expected", expected1)

      var pass1 = true
      for (r <- 0 until n; c <- 0 until n) {
        if (C1(r)(c) != expected1(r)(c)) {
          println(s"  MISMATCH at ($r,$c): actual=${C1(r)(c)} expected=${expected1(r)(c)}")
          pass1 = false
        }
      }
      println(s"  Matmul 1: ${if (pass1) "PASS" else "FAIL"}")

      // === Matmul 2 ===
      println("\nW2:"); printMatrix("", W2)
      println("A2 = 2*identity")
      println("Expected C2 = A2*W2 = 2*W2")

      loadWeight(dut, W2, prop = 1, "W2")
      val C2 = compute(dut, A2, prop = 0, "A2*W2")
      val expected2 = matmul(A2, W2)
      printMatrixL("C2 actual", C2)
      printMatrix("C2 expected", expected2)

      var pass2 = true
      for (r <- 0 until n; c <- 0 until n) {
        if (C2(r)(c) != expected2(r)(c)) {
          println(s"  MISMATCH at ($r,$c): actual=${C2(r)(c)} expected=${expected2(r)(c)}")
          pass2 = false
        }
      }
      println(s"  Matmul 2: ${if (pass2) "PASS" else "FAIL"}")

      assert(pass1, "Matmul 1 failed")
      assert(pass2, "Matmul 2 failed")
      println("\n=== ALL PASS ===")
    }
  }
}
