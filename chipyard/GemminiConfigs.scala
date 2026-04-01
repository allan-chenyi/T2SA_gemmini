package chipyard

import org.chipsalliance.cde.config.Config

// ------------------------------
// Configs with Gemmini RoCC
// ------------------------------

// DOC include start: GemminiRocketConfig
class GemminiRocketConfig extends Config(
  new gemmini.DefaultGemminiConfig ++                            // use Gemmini systolic array GEMM accelerator
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)
// DOC include end: GemminiRocketConfig

class FPGemminiRocketConfig extends Config(
  new gemmini.GemminiFP32DefaultConfig ++                         // use FP32Gemmini systolic array GEMM accelerator
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

class LeanGemminiRocketConfig extends Config(
  new gemmini.LeanGemminiConfig ++                                 // use Lean Gemmini systolic array GEMM accelerator
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

class LeanGemminiPrintfRocketConfig extends Config(
  new gemmini.LeanGemminiPrintfConfig ++                                 // use Lean Gemmini systolic array GEMM accelerator
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

class ReRoCCManyGemminiConfig extends Config(
  new rerocc.WithReRoCC ++
  new gemmini.LeanGemminiConfig ++                              // rerocc tile3 is gemmini
  new gemmini.LeanGemminiConfig ++                              // rerocc tile2 is gemmini
  new gemmini.LeanGemminiConfig ++                              // rerocc tile1 is gemmini
  new gemmini.LeanGemminiConfig ++                              // rerocc tile0 is gemmini
  new freechips.rocketchip.rocket.WithNHugeCores(4) ++           // 4 rocket cores
  new chipyard.config.AbstractConfig)

// --- 8x8 PPA Comparison Configs ---
class Baseline8x8WSRocketConfig extends Config(
  new gemmini.Baseline8x8WSGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

class Baseline8x8OSRocketConfig extends Config(
  new gemmini.Baseline8x8OSGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

class TwistWSSingleOpRocketConfig extends Config(
  new gemmini.TwistWSSingleOpGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

class TwistWSDualOpRocketConfig extends Config(
  new gemmini.TwistWSDualOpGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

// --- 16x16 PPA Comparison Configs ---
class Baseline16x16WSRocketConfig extends Config(
  new gemmini.Baseline16x16WSGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)
class Baseline16x16OSRocketConfig extends Config(
  new gemmini.Baseline16x16OSGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)
class Twist16x16WSSingleOpRocketConfig extends Config(
  new gemmini.Twist16x16WSSingleOpGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

// --- 32x32 PPA Comparison Configs (dma_buswidth=256) ---
class Baseline32x32WSRocketConfig extends Config(
  new gemmini.Baseline32x32WSGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new chipyard.config.AbstractConfig)
class Baseline32x32OSRocketConfig extends Config(
  new gemmini.Baseline32x32OSGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new chipyard.config.AbstractConfig)
class Twist32x32WSSingleOpRocketConfig extends Config(
  new gemmini.Twist32x32WSSingleOpGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new chipyard.config.AbstractConfig)

// --- 64x64 PPA Comparison Configs (dma_buswidth=512) ---
class Baseline64x64WSRocketConfig extends Config(
  new gemmini.Baseline64x64WSGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(512) ++
  new chipyard.config.AbstractConfig)
class Baseline64x64OSRocketConfig extends Config(
  new gemmini.Baseline64x64OSGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(512) ++
  new chipyard.config.AbstractConfig)
class Twist64x64WSSingleOpRocketConfig extends Config(
  new gemmini.Twist64x64WSSingleOpGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(512) ++
  new chipyard.config.AbstractConfig)

// --- 128x128 PPA Comparison Configs (dma_buswidth=1024, sbus=512 max due to TL constraint) ---
// SystemBusWidth capped at 512 (cacheBlockBytes=64B=512b). TLWidthWidget adapts 1024→512.
class Baseline128x128WSRocketConfig extends Config(
  new gemmini.Baseline128x128WSGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(512) ++
  new chipyard.config.AbstractConfig)
class Baseline128x128OSRocketConfig extends Config(
  new gemmini.Baseline128x128OSGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(512) ++
  new chipyard.config.AbstractConfig)
class Twist128x128WSSingleOpRocketConfig extends Config(
  new gemmini.Twist128x128WSSingleOpGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(512) ++
  new chipyard.config.AbstractConfig)

class GemminiShuttleConfig extends Config(
  new gemmini.DefaultGemminiConfig ++                            // use Gemmini systolic array GEMM accel
  new shuttle.common.WithNShuttleCores ++
  new chipyard.config.AbstractConfig)
