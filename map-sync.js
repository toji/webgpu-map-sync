// Copyright 2022 Brandon Jones
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Disclaimer: Everything about this is awful, and I don't recommend anybody uses it. It exists to
// demonstrate that this type of operation can already be done without explicit API support, for
// better or for worse.

// WebGPU canvas used to render data into prior to copying to WebGL.
const gpuCanvas = new OffscreenCanvas(1, 1);
const gpuCtx = gpuCanvas.getContext('webgpu');
const gpuUniformArray = new Uint32Array(2);

// WebGL2 context used to read data out of the WebGPU canvas in order to do gl.readPixels.
const glCanvas = new OffscreenCanvas(1, 1);
const gl = glCanvas.getContext('webgl2');
const texture = gl.createTexture();
const framebuffer = gl.createFramebuffer();

gl.bindTexture(gl.TEXTURE_2D, texture);
gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

// Private property caches for various externally provided WebGPU objects
const deviceReadbackHelpers = new WeakMap();
const readSyncBufferState = new WeakMap();

// Custom usage flag to use in place of MAP_READ when sync readback is desired.
GPUBufferUsage.MAP_READ_SYNC = 0x1000;

// Override create buffer to track the sizes of buffers that use the MAP_READ_SYNC usage.
const originalCreateBuffer = GPUDevice.prototype.createBuffer;
GPUDevice.prototype.createBuffer = function(descriptor) {
  let readSync = false;

  // Just like MAP_READ, COPY_DST is the only valid flag to combine with MAP_READ_SYNC. If anything
  // else is used the MAP_READ_SYNC flag won't be stripped out and the buffer creation will fail.
  if (descriptor.usage == GPUBufferUsage.MAP_READ_SYNC ||
      descriptor.usage == (GPUBufferUsage.MAP_READ_SYNC | GPUBufferUsage.COPY_DST)) {
    // If the buffer was created with our custom MAP_READ_SYNC flag, mark it as read-sync-able and
    // replace it with STORAGE usage.
    descriptor.usage &= ~0x1000;
    descriptor.usage |= GPUBufferUsage.STORAGE;
    readSync = true;
  }

  const buffer = originalCreateBuffer.call(this, descriptor);

  if (readSync) {
    readSyncBufferState.set(buffer, {
      size: descriptor.size,
      device: this,
      mapped: false,
    });
  }

  return buffer;
}

// Add a new method to buffers to do a synchronous readback if they were created with MAP_READ_SYNC.
GPUBuffer.prototype.mapSync = function(mode) {
  if (mode != GPUMapMode.READ) {
    // TODO: Should be a validation error.
    throw new Error('mapSync only supports READ mode.');
  }

  const readSyncState = readSyncBufferState.get(this);
  if (readSyncState === undefined) {
    // TODO: Should be a validation error.
    throw new Error('Buffer was not created with MAP_READ_SYNC usage.');
  }

  if (readSyncState.mapped === true) {
    // TODO: Should be a validation error.
    throw new Error('Buffer is already mapped.');
  }

  // Mark as mapped, but don't do the actual work until calling getMappedRange(),
  // because otherwise we might read back the entire buffer only to find that
  // developer only wanted a small part of it.
  readSyncState.mapped = true;

  return bufferReadSync(readSyncState.device, this, 0, readSyncState.size);
}

// Override getMappedRange to add custom behavior for MAP_READ_SYNC buffers
const originalGetMappedRange = GPUBuffer.prototype.getMappedRange;
GPUBuffer.prototype.getMappedRange = function(offset, size) {
  const readSyncState = readSyncBufferState.get(this);
  if (readSyncState !== undefined && readSyncState.mapped === true) {
    offset = offset || 0;
    size = size || readSyncState.size - offset;
    // TODO: For maximum accuracy, should track mapped ranges and esure they don't overlap here.
    return bufferReadSync(readSyncState.device, this, offset, size);
  }

  return originalGetMappedRange.call(this, offset, size);
}

// Override unmap to add custom behavior for MAP_READ_SYNC buffers
const originalUnmap = GPUBuffer.prototype.unmap;
GPUBuffer.prototype.unmap = function(offset, size) {
  const readSyncState = readSyncBufferState.get(this);
  if (readSyncState !== undefined && readSyncState.mapped === true) {
    readSyncState.mapped = false;
    // TODO: Should track the buffers returned by getMappedRange and destroy them here.
  }

  return originalUnmap.call(this);
}

function getDeviceReadbackHelpers(device) {
  if (deviceReadbackHelpers.has(device)) {
    return deviceReadbackHelpers.get(device);
  }

  // Shader which reads content out of a buffer and packs it into an RGBA texture.
  const shaderModule = device.createShaderModule({
    code: `
      var<private> pos : array<vec2<f32>, 3> = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(-1.0, 3.0), vec2<f32>(3.0, -1.0));

      struct VertexOutput {
        @builtin(position) position : vec4<f32>;
        @location(0) texCoord : vec2<f32>;
      };

      @stage(vertex)
      fn vertMain(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
        var output : VertexOutput;
        output.texCoord = pos[vertexIndex] * vec2<f32>(0.5, 0.5) + vec2<f32>(0.5);
        output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
        return output;
      }

      struct InputBuffer {
        bytes : array<u32>;
      };
      @group(0) @binding(0) var<storage> input : InputBuffer;
      @group(0) @binding(1) var<uniform> canvasSize : vec2<u32>;

      @stage(fragment)
      fn fragMain(@location(0) texCoord : vec2<f32>) -> @location(0) vec4<f32> {
        let index : u32 = u32(texCoord.x * f32(canvasSize.x)) +
                          u32(texCoord.y * f32(canvasSize.y)) * canvasSize.x;
        let bytes : u32 = input.bytes[index];

        return vec4(
          f32(bytes & 0xffu) / 255.0,
          f32((bytes >> 8u) & 0xffu) / 255.0,
          f32((bytes >> 16u) & 0xffu) / 255.0,
          f32((bytes >> 24u) & 0xffu) / 255.0
        );
      }
    `,
  });

  const pipeline = device.createRenderPipeline({
    vertex: {
      module: shaderModule,
      entryPoint: 'vertMain',
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fragMain',
      targets: [{
        format: 'bgra8unorm',
      }],
    }
  });

  const uniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  const readbackHelpers = {
    pipeline,
    uniformBuffer,
  };

  deviceReadbackHelpers.set(device, readbackHelpers);

  return readbackHelpers;
}

// Reads back the given buffer range and
function bufferReadSync(device, buffer, offset, size) {
  // Storage offsets have to be aligned at 256 byte boundaries.
  const alignedOffset = Math.floor(offset / 256) * 256;
  const alignedSize = size + (offset - alignedOffset);

  // TODO: Find a better equation for this
  const pixelSize = Math.ceil(alignedSize / 4);
  const width = Math.min(1024, pixelSize);
  const height = Math.ceil(pixelSize / width);

  // Resize the WebGPU canvas
  gpuCanvas.width = width;
  gpuCanvas.height = height;
  gpuCtx.configure({
    device,
    format: 'bgra8unorm',
  });

  // Draw the buffer contents to the WebGPU canvas
  const readbackHelpers = getDeviceReadbackHelpers(device);

  gpuUniformArray[0] = width;
  gpuUniformArray[1] = height;

  device.queue.writeBuffer(readbackHelpers.uniformBuffer, 0, gpuUniformArray);

  const bindGroup = device.createBindGroup({
    layout: readbackHelpers.pipeline.getBindGroupLayout(0),
    entries: [{
      binding: 0,
      resource: {
        buffer: buffer,
        offset: alignedOffset,
        size: alignedSize,
      },
    }, {
      binding: 1,
      resource: {
        buffer: readbackHelpers.uniformBuffer,
      },
    }],
  });

  const outputTexture = gpuCtx.getCurrentTexture();

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginRenderPass({
    colorAttachments: [{
      view: outputTexture.createView(),
      loadOp: 'clear',
      storeOp: 'store'
    }]
  });

  passEncoder.setPipeline(readbackHelpers.pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.draw(3);

  passEncoder.end();

  device.queue.submit([commandEncoder.finish()]);

  // Copy the WebGPU canvas to a WebGL framebuffer and read back the pixels.
  const readbackSize = width * height * 4;
  let bufferData = new Uint8Array(readbackSize);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, gpuCanvas);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, bufferData);

  // Return the array buffer for the readPixels data.
  if (alignedOffset != offset || readbackSize != size) {
    // In some cases it needs to be adjusted to account for differences in either the readback
    // buffer size or the alignment offset. In those cases we return a copy of a subsection of the
    // buffer instead.
    const sliceOffset = offset - alignedOffset;
    return bufferData.slice(sliceOffset, alignedSize).buffer;
  }

  return bufferData.buffer;
}