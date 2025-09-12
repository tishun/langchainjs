const { TestEnvironment } = require("jest-environment-node");

class AdjustedTestEnvironmentToSupportFloat32Array extends TestEnvironment {
  constructor(config, context) {
    // Make `instanceof Float32Array` return true in tests
    // to avoid https://github.com/xenova/transformers.js/issues/57 and https://github.com/jestjs/jest/issues/2549
    super(config, context);
    this.global.Float32Array = Float32Array;

    // Add ReadableStream polyfill for Node.js environments that don't have it
    if (typeof this.global.ReadableStream === "undefined") {
      try {
        const { ReadableStream } = require("stream/web");
        this.global.ReadableStream = ReadableStream;
      } catch (e) {
        // Fallback for older Node.js versions
        try {
          const { ReadableStream } = require("web-streams-polyfill");
          this.global.ReadableStream = ReadableStream;
        } catch (e2) {
          // If no polyfill is available, create a minimal mock
          this.global.ReadableStream = class ReadableStream {};
        }
      }
    }
  }
}

module.exports = AdjustedTestEnvironmentToSupportFloat32Array;
