/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/* eslint-disable no-undef */
// Load Emscripten Module, need to change path to root/lib
const path = require("path");
const fs = require("fs");
const assert = require("assert");
const tvmjs = require("../../dist");
const execSync = require('child_process').execSync;


const wasmPath = tvmjs.wasmPath();
const EmccWASI = require(path.join(wasmPath, "tvmjs_runtime.wasi.js"));

const wasmBinaryPath = path.join(wasmPath, "test_addone.wasm")
const wasmSource = fs.readFileSync(wasmBinaryPath);

const tvm = new tvmjs.Instance(
  new WebAssembly.Module(wasmSource),
  new EmccWASI()
);

// Load system library
const sysLib = tvm.systemLib();

function randomArray(length, max) {
  return Array.apply(null, Array(length)).map(function () {
    return Math.random() * max;
  });
}

test("add one", () => {
  // grab pre-loaded function
  const faddOne = sysLib.getFunction("add_one");
  assert(tvm.isPackedFunc(faddOne));
  const n = 124;
  const A = tvm.empty(n).copyFrom(randomArray(n, 1));
  const B = tvm.empty(n);
  // call the function.
  faddOne(A, B);
  const AA = A.toArray(); // retrieve values in js array
  const BB = B.toArray(); // retrieve values in js array
  // verify
  for (var i = 0; i < BB.length; ++i) {
    assert(Math.abs(BB[i] - (AA[i] + 1)) < 1e-5);
  }
  faddOne.dispose();
});
