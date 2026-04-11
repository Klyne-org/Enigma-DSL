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

@_cdecl("enigma_release")
public func enigma_release(_ ptr: UnsafeMutableRawPointer) {
    Unmanaged<AnyObject>.fromOpaque(ptr).release()
}
