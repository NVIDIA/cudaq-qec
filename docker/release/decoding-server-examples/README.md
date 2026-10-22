# CUDA-Q QEC decoding server: examples

This image ships `decoding_server` and everything needed to write, compile and
run CUDA-Q programs that decode against it. The programs in this directory are
the realtime-decoding examples from the CUDA-Q QEC test suite, included as a
starting point.

## Layout

| Path | Contents |
| --- | --- |
| `~/examples/` | Five surface-code memory experiments with realtime decoding, plus `build.sh` |
| `/opt/nvidia/cudaqx/bin` | `decoding_server`, `playback-emulator`, `hsb_fpga_syndrome_playback`, example server configs |

`nvq++`, the CUDA-Q and QEC headers, and all libraries are on the default
search paths. A program is compiled with `nvq++ --target stim`, linking the QEC
libraries it needs, and runs without further setup. `build.sh` shows the exact
flags.

## Build

    cd ~/examples
    ./build.sh            # all examples, into ./build
    ./build.sh surface_code-1   # one example

Each example is built in two variants where that applies: `<name>-local`
decodes in-process, and `<name>-cqr` has its kernel lowered with
`-frealtime-lowering` so every decoding call goes over the realtime channel to
a separately running `decoding_server`.

## Running the examples

All commands assume `cd ~/examples` after `./build.sh`. The programs
print their options with `--help` (except surface_code-5, which takes only the
flags shown below).

**surface_code-1**: one realtime decoder per logical qubit, circuit-level
depolarizing noise on the stabilizer CNOTs. Decode in-process:

    ./build/surface_code-1-local --distance 3 --num_rounds 3 --num_shots 200 \
        --decoder_type pymatching

Against a separate server (the two-process flow): let the program write a server
config from its detector error model, start the server on it, then point the
lowered program at the server's port.

    ./build/surface_code-1-cqr --distance 3 --num_rounds 3 --num_shots 200 \
        --save_dem cfg.yml --decoder_type pymatching
    decoding_server --config=cfg.yml --transport=udp --port=0 &
    #   ...prints: QEC_DECODING_SERVER_READY port=<N> ...
    QEC_DECODING_SERVER_PORT=<N> QEC_DECODING_SERVER_TRANSPORT=udp \
    CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch \
        ./build/surface_code-1-cqr --distance 3 --num_rounds 3 --num_shots 200 \
            --load_dem cfg.yml --decoder_type pymatching

Syndrome replay: capture every syndrome and correction during a live run, then
replay the file through the decoders offline and check the corrections match.

    ./build/surface_code-1-local --distance 3 --num_rounds 3 --num_shots 200 \
        --save_syndrome syndromes.txt
    ./build/surface_code-1-local --distance 3 --num_rounds 3 \
        --load_syndrome syndromes.txt

**surface_code-2**: the detector error model is characterized from in-kernel
SPAM noise via `cudaq::dem_from_kernel`. Two phases, write then load:

    ./build/surface_code-2-local --distance 3 --num_shots 100 --save_dem sc2.yml
    ./build/surface_code-2-local --distance 3 --num_shots 100 --load_dem sc2.yml

**surface_code-3**: two decoders per logical qubit (separate X and Z decoding),
in either basis. `--state_prep prep0` or `prepp` is required (not listed by
`--help`); otherwise two phases as for surface_code-2:

    ./build/surface_code-3-local --distance 3 --num_shots 100 --state_prep prep0 --save_dem sc3.yml
    ./build/surface_code-3-local --distance 3 --num_shots 100 --state_prep prep0 --load_dem sc3.yml

**surface_code-4-yaml**: the decoder configuration crosses the process boundary
as a file. Phase one writes a YAML config from the DEM; phase two loads it.

    ./build/surface_code-4-yaml --distance 3 --num_rounds 3 --num_shots 200 \
        --save_dem sc4.yml --decoder_type pymatching
    ./build/surface_code-4-yaml --distance 3 --num_rounds 3 --num_shots 200 \
        --yaml sc4.yml

The `-cqr` variant decodes through an external server; start `decoding_server`
on `sc4.yml` as above and set `QEC_DECODING_SERVER_PORT` before running it.

**surface_code-5-per-decoder-rings**: one ring buffer and one dispatcher per
decoder, via `cudaq::device_call`'s device-id overload. Self-checking.

    CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch \
        ./build/surface_code-5-per-decoder-rings --distance 3 --num_rounds 3 --num_shots 100

## Hardware path (ConnectX NIC required)

The cpu_roce transport moves syndromes from an FPGA, or from the software
emulator `hsb_fpga_emulator` (in `/opt/nvidia/cudaq-realtime/bin`), into the
server over RDMA. It needs an RDMA-capable NIC passed into the container
(`--device /dev/infiniband`, typically `--privileged`) and cannot run without
one. In outline:

1. Start the server on the RoCE transport with the FPGA handshake, naming the
   NIC and the FPGA's queue pair. Provider arguments use `--flag=value` form
   and are passed through to the transport:

       decoding_server --config=cfg.yml --transport=cpu-roce --qp_config=hsb_fpga \
           --device=<mlx5_N> --peer-ip=<fpga-ip> --remote-qp=<qp> \
           --num-slots=<n> --slot-size=<bytes> --frame-size=<bytes>

   On startup it prints its own RDMA endpoint (queue pair, rkey, buffer
   address).

2. Program the FPGA, or the emulator, with that endpoint and stream the
   syndromes:

       hsb_fpga_syndrome_playback --hsb-ip <fpga-ip> --data-dir <dir> \
           --qp-number <qp> --rkey <rkey> --buffer-addr <addr> --verify

   `hsb_fpga_syndrome_playback --help` lists the remaining options.

Network setup and the full procedure are documented in the CUDA-Q QEC docs under
the realtime decoding examples.

## Decoder plugins

The built-in decoders (`multi_error_lut`, `single_error_lut`, `sliding_window`
and others) live in `libcudaq-qec-decoders.so`. Plugin decoders are loaded from
`/opt/nvidia/cudaqx/lib/decoder-plugins`; this image ships `pymatching`,
`chromobius`, `trt_decoder` and a small `example` plugin. The `nv-qldpc-decoder`
GPU plugin is present when the image was built with the proprietary artifacts.
