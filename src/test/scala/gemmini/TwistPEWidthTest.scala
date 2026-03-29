package gemmini

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/** Minimal test to check TwistPE output width with negative values */
class TwistPEWidthSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "TwistPE width"

  it should "output correct negative values" in {
    test(new TwistPE(SInt(8.W), SInt(8.W), SInt(20.W), SInt(32.W))) { dut =>

      dut.reset.poke(true.B); dut.clock.step(1)
      dut.reset.poke(false.B); dut.clock.step(1)

      // Load weight -7 into buffer2 (prop=0)
      dut.io.in_valid.poke(true.B)
      dut.io.in_propagate.poke(0.U)
      dut.io.in_weight.poke((-7).S)
      dut.io.in_lock.poke(true.B)
      dut.io.in_ifmap.poke(0.S)
      dut.io.in_psum.poke(0.S)
      dut.clock.step(1)

      // Let capture happen
      dut.io.in_lock.poke(false.B)
      dut.io.in_weight.poke(0.S)
      dut.clock.step(1)

      // Check buffer2
      val buf2 = dut.io.debug_buffer2.peek().litValue
      val buf2w = dut.io.debug_buffer2.getWidth
      println(s"buffer2: raw=$buf2, width=$buf2w")

      // Now compute: prop=1, ifmap=1, psum=0
      dut.io.in_propagate.poke(1.U)
      dut.io.in_ifmap.poke(1.S)
      dut.io.in_psum.poke(0.S)

      val outRaw = dut.io.out_psum.peek().litValue
      val outW = dut.io.out_psum.getWidth
      val wsel = dut.io.debug_weight_sel.peek().litValue
      val wselW = dut.io.debug_weight_sel.getWidth
      println(s"out_psum: raw=$outRaw, width=$outW")
      println(s"weight_sel: raw=$wsel, width=$wselW")

      // Sign-interpret
      def signInterp(v: BigInt, w: Int): Long = {
        if (v.testBit(w - 1)) (v - (BigInt(1) << w)).toLong else v.toLong
      }
      println(s"out_psum signed: ${signInterp(outRaw, outW)}")
      println(s"weight_sel signed: ${signInterp(wsel, wselW)}")
      println(s"buffer2 signed: ${signInterp(buf2, buf2w)}")

      // Test with positive weight too
      dut.io.in_propagate.poke(0.U)
      dut.io.in_weight.poke(7.S)
      dut.io.in_lock.poke(true.B)
      dut.clock.step(1)
      dut.io.in_lock.poke(false.B)
      dut.io.in_weight.poke(0.S)
      dut.clock.step(1)

      dut.io.in_propagate.poke(1.U)
      dut.io.in_ifmap.poke(1.S)
      val outRaw2 = dut.io.out_psum.peek().litValue
      println(s"\nPositive weight 7: out_psum raw=$outRaw2, signed=${signInterp(outRaw2, outW)}")

      // Test: ifmap=1, weight=-7, psum=-3
      dut.io.in_propagate.poke(1.U)
      dut.io.in_psum.poke((-3).S)
      dut.io.in_ifmap.poke(1.S)
      // Need buffer2 to have -7 again, reload
      dut.io.in_propagate.poke(0.U)
      dut.io.in_weight.poke((-7).S)
      dut.io.in_lock.poke(true.B)
      dut.clock.step(1)
      dut.io.in_lock.poke(false.B)
      dut.io.in_weight.poke(0.S)
      dut.clock.step(1)

      dut.io.in_propagate.poke(1.U)
      dut.io.in_ifmap.poke(1.S)
      dut.io.in_psum.poke((-3).S)
      val outRaw3 = dut.io.out_psum.peek().litValue
      println(s"\nifmap=1, weight=-7, psum=-3: out_psum raw=$outRaw3, width=$outW, signed=${signInterp(outRaw3, outW)}")
      println(s"Expected: 1*(-7)+(-3) = -10")
    }
  }
}
