
# python tutorials/frontend/from_paddle_lmk_wasm.py

cp /data/lmk_demo/lmk_demo/data/mv6s_shift_aug_infer/lmk.json ./
cp /data/lmk_demo/lmk_demo/data/mv6s_shift_aug_infer/lmk.wasm ./
cp /data/lmk_demo/lmk_demo/data/mv6s_shift_aug_infer/lmk.params ./

cp /workspace/tvm/web/dist/wasm/tvmjs_runtime.wasi.js /workspace/tvm/web/deploy
cp /workspace/tvm/web/dist/tvmjs.bundle.js /workspace/tvm/web/deploy


python3 -m http.server 8080 --bind 0.0.0.0 -d ./ 
# python3 nocache_server.py 8080 --bind 0.0.0.0 -d ./ 
