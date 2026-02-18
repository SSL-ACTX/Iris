/* index.js */
const { existsSync, readFileSync } = require('fs')
const { join } = require('path')

const { platform, arch } = process

let nativeBinding = null
let localFileExisted = false
let loadError = null

function isMusl() {
  // Simple check for Musl/Alpine Linux
  try {
    return readFileSync('/usr/bin/ldd', 'utf8').includes('musl')
  } catch (e) {
    return true
  }
}

switch (platform) {
  case 'android':
    if (arch === 'arm64') {
      try {
        nativeBinding = require('./myrmidon.android-arm64.node')
      } catch (e) {
        loadError = e
      }
    } else if (arch === 'arm') {
      try {
        nativeBinding = require('./myrmidon.android-arm-eabi.node')
      } catch (e) {
        loadError = e
      }
    }
    break
  case 'win32':
    if (arch === 'x64') {
      try {
        nativeBinding = require('./myrmidon.win32-x64-msvc.node')
      } catch (e) {
        loadError = e
      }
    }
    break
  case 'darwin':
    try {
        nativeBinding = require('./myrmidon.darwin-universal.node')
    } catch (e) {
        // Fallback for specific archs if universal not found
        if (arch === 'x64') {
             nativeBinding = require('./myrmidon.darwin-x64.node')
        } else if (arch === 'arm64') {
             nativeBinding = require('./myrmidon.darwin-arm64.node')
        }
    }
    break
  case 'linux':
    if (arch === 'x64') {
      if (isMusl()) {
        try {
          nativeBinding = require('./myrmidon.linux-x64-musl.node')
        } catch (e) {
          loadError = e
        }
      } else {
        try {
          nativeBinding = require('./myrmidon.linux-x64-gnu.node')
        } catch (e) {
          loadError = e
        }
      }
    } else if (arch === 'arm64') {
       if (isMusl()) {
        try {
          nativeBinding = require('./myrmidon.linux-arm64-musl.node')
        } catch (e) {
          loadError = e
        }
      } else {
        try {
          nativeBinding = require('./myrmidon.linux-arm64-gnu.node')
        } catch (e) {
          loadError = e
        }
      }
    }
    break
  default:
    throw new Error(`Unsupported OS: ${platform}, architecture: ${arch}`)
}

if (!nativeBinding) {
  if (loadError) {
    throw loadError
  }
  throw new Error(`Failed to load native binding`)
}

const { NodeRuntime, JsMailbox, WrappedMessage, JsSystemMessage } = nativeBinding

module.exports.NodeRuntime = NodeRuntime
module.exports.JsMailbox = JsMailbox
module.exports.WrappedMessage = WrappedMessage
module.exports.JsSystemMessage = JsSystemMessage