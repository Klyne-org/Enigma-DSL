// libenigma_runtime.swift
// Minimal Metal compute runtime with C-callable API.
// Build: swiftc -O -emit-library -o libenigma_runtime.dylib libenigma_runtime.swift

import Metal
import Foundation

// ── Device & Queue ───────────────────────────────────────────────────────

@_cdecl("enigma_create_device")
public func enigma_create_device() -> UnsafeMutableRawPointer? {
    guard let device = MTLCreateSystemDefaultDevice() else {
        fputs("enigma: no Metal device found\n", stderr)
        return nil
    }
    return Unmanaged<MTLDevice>.passRetained(device).toOpaque()
}

@_cdecl("enigma_create_queue")
public func enigma_create_queue(_ devicePtr: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer? {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    guard let queue = device.makeCommandQueue() else {
        fputs("enigma: failed to create command queue\n", stderr)
        return nil
    }
    return Unmanaged<MTLCommandQueue>.passRetained(queue).toOpaque()
}

// ── Library & Pipeline ───────────────────────────────────────────────────

@_cdecl("enigma_load_library")
public func enigma_load_library(
    _ devicePtr: UnsafeMutableRawPointer,
    _ path: UnsafePointer<CChar>
) -> UnsafeMutableRawPointer? {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    let url = URL(fileURLWithPath: String(cString: path))
    do {
        let library = try device.makeLibrary(URL: url)
        return Unmanaged<MTLLibrary>.passRetained(library).toOpaque()
    } catch {
        fputs("enigma: failed to load metallib: \(error)\n", stderr)
        return nil
    }
}

@_cdecl("enigma_load_library_from_data")
public func enigma_load_library_from_data(
    _ devicePtr: UnsafeMutableRawPointer,
    _ data: UnsafeRawPointer,
    _ len: Int
) -> UnsafeMutableRawPointer? {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    let dispatchData = DispatchData(
        bytesNoCopy: UnsafeRawBufferPointer(start: data, count: len),
        deallocator: .custom(nil, { })
    )
    do {
        let library = try device.makeLibrary(data: dispatchData as __DispatchData)
        return Unmanaged<MTLLibrary>.passRetained(library).toOpaque()
    } catch {
        fputs("enigma: failed to load metallib from data: \(error)\n", stderr)
        return nil
    }
}

@_cdecl("enigma_create_pipeline")
public func enigma_create_pipeline(
    _ devicePtr: UnsafeMutableRawPointer,
    _ libraryPtr: UnsafeMutableRawPointer,
    _ funcName: UnsafePointer<CChar>
) -> UnsafeMutableRawPointer? {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    let library = Unmanaged<MTLLibrary>.fromOpaque(libraryPtr).takeUnretainedValue()
    let name = String(cString: funcName)
    guard let function = library.makeFunction(name: name) else {
        fputs("enigma: function '\(name)' not found in library\n", stderr)
        return nil
    }
    do {
        let pso = try device.makeComputePipelineState(function: function)
        return Unmanaged<MTLComputePipelineState>.passRetained(pso).toOpaque()
    } catch {
        fputs("enigma: failed to create pipeline state: \(error)\n", stderr)
        return nil
    }
}

// ── Buffers ──────────────────────────────────────────────────────────────

@_cdecl("enigma_create_buffer")
public func enigma_create_buffer(
    _ devicePtr: UnsafeMutableRawPointer,
    _ data: UnsafeRawPointer,
    _ len: Int
) -> UnsafeMutableRawPointer? {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    guard let buffer = device.makeBuffer(bytes: data, length: len, options: .storageModeShared) else {
        fputs("enigma: failed to create buffer (\(len) bytes)\n", stderr)
        return nil
    }
    return Unmanaged<MTLBuffer>.passRetained(buffer).toOpaque()
}

@_cdecl("enigma_create_buffer_empty")
public func enigma_create_buffer_empty(
    _ devicePtr: UnsafeMutableRawPointer,
    _ len: Int
) -> UnsafeMutableRawPointer? {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    guard let buffer = device.makeBuffer(length: len, options: .storageModeShared) else {
        fputs("enigma: failed to create empty buffer (\(len) bytes)\n", stderr)
        return nil
    }
    return Unmanaged<MTLBuffer>.passRetained(buffer).toOpaque()
}

@_cdecl("enigma_buffer_contents")
public func enigma_buffer_contents(_ bufferPtr: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer {
    let buffer = Unmanaged<MTLBuffer>.fromOpaque(bufferPtr).takeUnretainedValue()
    return buffer.contents()
}

@_cdecl("enigma_buffer_length")
public func enigma_buffer_length(_ bufferPtr: UnsafeMutableRawPointer) -> Int {
    let buffer = Unmanaged<MTLBuffer>.fromOpaque(bufferPtr).takeUnretainedValue()
    return buffer.length
}

// ── Dispatch ─────────────────────────────────────────────────────────────

@_cdecl("enigma_dispatch")
public func enigma_dispatch(
    _ psoPtr: UnsafeMutableRawPointer,
    _ queuePtr: UnsafeMutableRawPointer,
    _ bufPtrs: UnsafePointer<UnsafeMutableRawPointer?>,
    _ bufCount: Int,
    _ gridX: Int, _ gridY: Int, _ gridZ: Int,
    _ threadsX: Int, _ threadsY: Int, _ threadsZ: Int
) -> Int32 {
    let pso = Unmanaged<MTLComputePipelineState>.fromOpaque(psoPtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()

    guard let cmdBuf = queue.makeCommandBuffer() else {
        fputs("enigma: failed to create command buffer\n", stderr)
        return -1
    }
    guard let encoder = cmdBuf.makeComputeCommandEncoder() else {
        fputs("enigma: failed to create compute encoder\n", stderr)
        return -1
    }

    encoder.setComputePipelineState(pso)

    for i in 0..<bufCount {
        if let rawPtr = bufPtrs[i] {
            let buffer = Unmanaged<MTLBuffer>.fromOpaque(rawPtr).takeUnretainedValue()
            encoder.setBuffer(buffer, offset: 0, index: i)
        }
    }

    let gridSize = MTLSize(width: gridX, height: gridY, depth: gridZ)
    let threadgroupSize = MTLSize(width: threadsX, height: threadsY, depth: threadsZ)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    if let error = cmdBuf.error {
        fputs("enigma: GPU execution error: \(error)\n", stderr)
        return -2
    }
    return 0
}

// ── Cleanup ──────────────────────────────────────────────────────────────

@_cdecl("enigma_dispatch_timed")
public func enigma_dispatch_timed(
    _ psoPtr: UnsafeMutableRawPointer,
    _ queuePtr: UnsafeMutableRawPointer,
    _ bufPtrs: UnsafePointer<UnsafeMutableRawPointer?>,
    _ bufCount: Int,
    _ gridX: Int, _ gridY: Int, _ gridZ: Int,
    _ threadsX: Int, _ threadsY: Int, _ threadsZ: Int,
    _ outGpuTimeUs: UnsafeMutablePointer<Double>
) -> Int32 {
    let pso = Unmanaged<MTLComputePipelineState>.fromOpaque(psoPtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()

    guard let cmdBuf = queue.makeCommandBuffer() else { return -1 }
    guard let encoder = cmdBuf.makeComputeCommandEncoder() else { return -1 }

    encoder.setComputePipelineState(pso)
    for i in 0..<bufCount {
        if let rawPtr = bufPtrs[i] {
            let buffer = Unmanaged<MTLBuffer>.fromOpaque(rawPtr).takeUnretainedValue()
            encoder.setBuffer(buffer, offset: 0, index: i)
        }
    }

    encoder.dispatchThreads(
        MTLSize(width: gridX, height: gridY, depth: gridZ),
        threadsPerThreadgroup: MTLSize(width: threadsX, height: threadsY, depth: threadsZ))
    encoder.endEncoding()

    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    outGpuTimeUs.pointee = (cmdBuf.gpuEndTime - cmdBuf.gpuStartTime) * 1_000_000
    return cmdBuf.error != nil ? -2 : 0
}

// ── Profiled dispatch (stage-boundary counter sampling) ─────────────────

private func enigmaTimestampCounterSet(_ device: MTLDevice) -> MTLCounterSet? {
    return device.counterSets?.first {
        $0.name.caseInsensitiveCompare(MTLCommonCounterSet.timestamp.rawValue) == .orderedSame
    }
}

@_cdecl("enigma_supports_stage_counters")
public func enigma_supports_stage_counters(_ devicePtr: UnsafeMutableRawPointer) -> Int32 {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    guard device.supportsCounterSampling(.atStageBoundary) else { return 0 }
    return enigmaTimestampCounterSet(device) != nil ? 1 : 0
}

// outTimings layout (all microseconds except [4]):
//   [0] command-buffer GPU time     (gpuEndTime - gpuStartTime)
//   [1] CPU scheduling time         (kernelEndTime - kernelStartTime)
//   [2] queue wait                  (gpuStartTime - kernelEndTime, clamped >= 0)
//   [3] encoder GPU time            (stage-boundary counter samples; 0 if unavailable)
//   [4] 1.0 if counter samples resolved, else 0.0
@_cdecl("enigma_dispatch_profiled")
public func enigma_dispatch_profiled(
    _ devicePtr: UnsafeMutableRawPointer,
    _ psoPtr: UnsafeMutableRawPointer,
    _ queuePtr: UnsafeMutableRawPointer,
    _ bufPtrs: UnsafePointer<UnsafeMutableRawPointer?>,
    _ bufCount: Int,
    _ gridX: Int, _ gridY: Int, _ gridZ: Int,
    _ threadsX: Int, _ threadsY: Int, _ threadsZ: Int,
    _ outTimings: UnsafeMutablePointer<Double>
) -> Int32 {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    let pso = Unmanaged<MTLComputePipelineState>.fromOpaque(psoPtr).takeUnretainedValue()
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    for i in 0..<5 { outTimings[i] = 0 }

    var sampleBuffer: MTLCounterSampleBuffer? = nil
    if device.supportsCounterSampling(.atStageBoundary),
       let counterSet = enigmaTimestampCounterSet(device) {
        let desc = MTLCounterSampleBufferDescriptor()
        desc.counterSet = counterSet
        desc.storageMode = .shared
        desc.sampleCount = 2
        sampleBuffer = try? device.makeCounterSampleBuffer(descriptor: desc)
    }

    guard let cmdBuf = queue.makeCommandBuffer() else { return -1 }

    let encoder: MTLComputeCommandEncoder?
    if let sb = sampleBuffer {
        let passDesc = MTLComputePassDescriptor()
        let attachment = passDesc.sampleBufferAttachments[0]!
        attachment.sampleBuffer = sb
        attachment.startOfEncoderSampleIndex = 0
        attachment.endOfEncoderSampleIndex = 1
        encoder = cmdBuf.makeComputeCommandEncoder(descriptor: passDesc)
    } else {
        encoder = cmdBuf.makeComputeCommandEncoder()
    }
    guard let enc = encoder else { return -1 }

    enc.setComputePipelineState(pso)
    for i in 0..<bufCount {
        if let rawPtr = bufPtrs[i] {
            let buffer = Unmanaged<MTLBuffer>.fromOpaque(rawPtr).takeUnretainedValue()
            enc.setBuffer(buffer, offset: 0, index: i)
        }
    }
    enc.dispatchThreads(
        MTLSize(width: gridX, height: gridY, depth: gridZ),
        threadsPerThreadgroup: MTLSize(width: threadsX, height: threadsY, depth: threadsZ))
    enc.endEncoding()

    // GPU-tick -> wall-time correlation, sampled around the measured region.
    let before = device.sampleTimestamps()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()
    let after = device.sampleTimestamps()

    outTimings[0] = (cmdBuf.gpuEndTime - cmdBuf.gpuStartTime) * 1_000_000
    outTimings[1] = (cmdBuf.kernelEndTime - cmdBuf.kernelStartTime) * 1_000_000
    outTimings[2] = max(0, (cmdBuf.gpuStartTime - cmdBuf.kernelEndTime) * 1_000_000)

    if let sb = sampleBuffer,
       let data = try? sb.resolveCounterRange(0..<2),
       data.count >= 2 * MemoryLayout<MTLCounterResultTimestamp>.stride {
        let samples = data.withUnsafeBytes {
            Array($0.bindMemory(to: MTLCounterResultTimestamp.self).prefix(2))
        }
        let startTs = samples[0].timestamp
        let endTs = samples[1].timestamp
        if startTs != MTLCounterErrorValue, endTs != MTLCounterErrorValue,
           endTs > startTs, after.gpu > before.gpu {
            let nsPerTick = Double(after.cpu - before.cpu) / Double(after.gpu - before.gpu)
            outTimings[3] = Double(endTs - startTs) * nsPerTick / 1_000.0
            outTimings[4] = 1.0
        }
    }

    return cmdBuf.error != nil ? -2 : 0
}

// ── Pipeline insight ─────────────────────────────────────────────────────

// outVals: [0] maxTotalThreadsPerThreadgroup
//          [1] threadExecutionWidth
//          [2] staticThreadgroupMemoryLength (bytes)
@_cdecl("enigma_pipeline_stats")
public func enigma_pipeline_stats(
    _ psoPtr: UnsafeMutableRawPointer,
    _ outVals: UnsafeMutablePointer<Int>
) {
    let pso = Unmanaged<MTLComputePipelineState>.fromOpaque(psoPtr).takeUnretainedValue()
    outVals[0] = pso.maxTotalThreadsPerThreadgroup
    outVals[1] = pso.threadExecutionWidth
    outVals[2] = pso.staticThreadgroupMemoryLength
}

// ── Programmatic Metal capture (.gputrace) ──────────────────────────────

// Returns 0 on success, -1 if .gputrace capture is unsupported (set
// MTL_CAPTURE_ENABLED=1), -2 if startCapture threw, -3 if no device.
@_cdecl("enigma_capture_start")
public func enigma_capture_start(_ path: UnsafePointer<CChar>) -> Int32 {
    let manager = MTLCaptureManager.shared()
    guard manager.supportsDestination(.gpuTraceDocument) else { return -1 }
    guard let device = MTLCreateSystemDefaultDevice() else { return -3 }
    let desc = MTLCaptureDescriptor()
    desc.captureObject = device
    desc.destination = .gpuTraceDocument
    desc.outputURL = URL(fileURLWithPath: String(cString: path))
    do {
        try manager.startCapture(with: desc)
        return 0
    } catch {
        fputs("enigma: capture start failed: \(error)\n", stderr)
        return -2
    }
}

@_cdecl("enigma_capture_stop")
public func enigma_capture_stop() {
    MTLCaptureManager.shared().stopCapture()
}

@_cdecl("enigma_release")
public func enigma_release(_ ptr: UnsafeMutableRawPointer) {
    Unmanaged<AnyObject>.fromOpaque(ptr).release()
}

// ── Device capability queries ────────────────────────────────────────────

@_cdecl("enigma_device_supports_family")
public func enigma_device_supports_family(
    _ devicePtr: UnsafeMutableRawPointer,
    _ familyRaw: Int32
) -> Int32 {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    guard let family = MTLGPUFamily(rawValue: Int(familyRaw)) else { return 0 }
    return device.supportsFamily(family) ? 1 : 0
}

@_cdecl("enigma_device_name")
public func enigma_device_name(
    _ devicePtr: UnsafeMutableRawPointer,
    _ outBuf: UnsafeMutablePointer<CChar>,
    _ bufLen: Int
) {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    let nameBytes = Array(device.name.utf8)
    let n = min(nameBytes.count, bufLen - 1)
    for i in 0..<n { outBuf[i] = CChar(bitPattern: nameBytes[i]) }
    outBuf[n] = 0
}

@_cdecl("enigma_device_max_threadgroup_memory")
public func enigma_device_max_threadgroup_memory(
    _ devicePtr: UnsafeMutableRawPointer
) -> Int {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    return device.maxThreadgroupMemoryLength
}

@_cdecl("enigma_device_max_threads_per_threadgroup")
public func enigma_device_max_threads_per_threadgroup(
    _ devicePtr: UnsafeMutableRawPointer
) -> Int {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    return device.maxThreadsPerThreadgroup.width
}

// ── Pipeline creation with function constants (Bug 3) ───────────────────
// Accepts `count` triples of (index, type_tag, value) where:
//   type_tag: 0=float, 1=int (i32), 2=uint (u32), 3=bool, 4=half
//   value:    stored as Double, cast by tag on the Swift side.

@_cdecl("enigma_create_pipeline_with_constants")
public func enigma_create_pipeline_with_constants(
    _ devicePtr: UnsafeMutableRawPointer,
    _ libraryPtr: UnsafeMutableRawPointer,
    _ funcName: UnsafePointer<CChar>,
    _ indices: UnsafePointer<Int32>,
    _ typeTags: UnsafePointer<Int32>,
    _ values: UnsafePointer<Double>,
    _ count: Int
) -> UnsafeMutableRawPointer? {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    let library = Unmanaged<MTLLibrary>.fromOpaque(libraryPtr).takeUnretainedValue()
    let name = String(cString: funcName)

    let constants = MTLFunctionConstantValues()
    for k in 0..<count {
        let idx = Int(indices[k])
        let tag = typeTags[k]
        let dv = values[k]
        switch tag {
        case 0:  // float
            var v: Float = Float(dv)
            constants.setConstantValue(&v, type: .float, index: idx)
        case 1:  // int32
            var v: Int32 = Int32(dv)
            constants.setConstantValue(&v, type: .int, index: idx)
        case 2:  // uint32
            var v: UInt32 = UInt32(dv)
            constants.setConstantValue(&v, type: .uint, index: idx)
        case 3:  // bool
            var v: Bool = dv != 0.0
            constants.setConstantValue(&v, type: .bool, index: idx)
        case 4:  // half
            // Metal wants 16-bit half; pass via Float16 on Apple Silicon.
            #if arch(arm64)
            var v: Float16 = Float16(dv)
            constants.setConstantValue(&v, type: .half, index: idx)
            #else
            var v: Float = Float(dv)
            constants.setConstantValue(&v, type: .float, index: idx)
            #endif
        default:
            fputs("enigma: unknown function_constant type tag \(tag)\n", stderr)
            return nil
        }
    }

    do {
        let function = try library.makeFunction(name: name, constantValues: constants)
        let pso = try device.makeComputePipelineState(function: function)
        return Unmanaged<MTLComputePipelineState>.passRetained(pso).toOpaque()
    } catch {
        fputs("enigma: failed to create specialized pipeline '\(name)': \(error)\n", stderr)
        return nil
    }
}
