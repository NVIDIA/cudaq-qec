AI Predecoding
==============

AI predecoding combines a trained neural network with a conventional decoder.
The network predicts corrections from detector events, leaving a residual
syndrome for the conventional decoder to resolve. Their predictions are
combined to produce the final logical correction. The aim is to reduce the
work left for the conventional decoder while maintaining or improving the
logical error rate of the complete pipeline.

CUDA-Q QEC provides TensorRT inference on the GPU and interfaces for connecting
it to a residual decoder. The examples use neural networks from
`NVIDIA Ising-Decoding <https://github.com/NVIDIA/Ising-Decoding>`__ with
PyMatching for surface codes or Chromobius for color codes. Each network must
match the circuit, detector layout, and noise assumptions of the experiment;
it is not a replacement for an arbitrary decoder configuration.

How the Stages Fit Together
---------------------------

1. **Prepare the input.** Convert circuit measurements into detector events
   using the measurement-to-detector mapping for the experiment.
2. **Run the predecoder.** Evaluate the neural network and use its predicted
   corrections to calculate the remaining detector events and its contribution
   to the logical correction.
3. **Decode the residual.** Pass those remaining events to the conventional
   decoder and combine the two stages' logical predictions.

For the color-code example, the Ising implementation and pretrained weights
are exported together with input conversion and output processing to ONNX.
The ``trt_decoder`` plugin builds a TensorRT engine from that export. Its
``observables_and_residual_detectors`` output format carries the predecoder's
logical flips and residual detectors; ``global_decoder="chromobius"`` selects
Chromobius to complete decoding. See the
:ref:`TensorRT Decoder API <trt_decoder_api_python>` for these options.

Choose an Example
-----------------

* :doc:`Comparing Ising and Chromobius for Color-Code Decoding
  </examples_rst/qec/color_code_demo>` compares Chromobius alone with the
  combined Ising and Chromobius pipeline. It replays sampled measurements
  through both configurations, in process and over UDP, to measure logical
  errors and runtime. This is a software experiment; it does not require a
  QPU or FPGA.
* :doc:`AI Predecoder with CUDA-Q Realtime
  </examples_rst/qec/realtime_predecoder_pymatching>` exercises a surface-code
  pipeline with a software injector, GPU predecoding, and a pool of CPU
  PyMatching decoders. Use it to study throughput, latency, and residual
  syndrome density under a controlled input rate.
* :doc:`AI Predecoder with CUDA-Q Realtime (with FPGA Data Injection)
  </examples_rst/qec/realtime_predecoder_fpga>` extends the surface-code
  pipeline to GPU RoCE input. It covers an FPGA emulator and real FPGA input;
  both configurations require the networking hardware listed in that guide.

The two surface-code demonstrations use the experimental C++
``realtime_pipeline`` and must be built from source. The color-code example
uses the ``trt_decoder`` plugin through the
:doc:`Playback Emulator <playback_emulator>` and a local
:doc:`Decoding Server <decoding_server>`. Their setup commands and timing
measurements are specific to those execution paths.

Evaluate the Complete Pipeline
------------------------------

Compare the conventional decoder alone and the combined pipeline on the same
sampled shots. Check the final logical corrections against known observables,
and measure runtime as well as logical error rate: fewer residual detector
events do not by themselves establish a faster or more accurate decoder.

Distinguish TensorRT inference time from the time taken by the full pipeline,
including residual decoding, scheduling, and communication. The color-code
example reports runtime amortized over all decoded rounds; the surface-code
benchmark reports request latency and throughput. Use each example's stated
measurement boundary when interpreting its results, and collect enough shots
to resolve the logical error rates you intend to compare.
