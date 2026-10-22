/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/
#pragma once
#include "cudaq/edem/composition.h"
#include "cudaq/qec/detector_error_model.h"

namespace cudaq::qec {
/// Convert a projected DEM to QEC's dense H/O and independent fault model.
/// Compose and project with cudaq-edem before calling this bridge. The C view
/// borrows its owner's storage for this call; the returned QEC model owns its
/// matrices and rates. Preserves the source model's fault column and
/// detector/observable row ordering.
/// Preserve every original column and its marginal probability. Exclusive
/// alternatives become independent Bernoulli mechanisms: this is a marginal
/// approximation, not an exact conversion of their joint distribution.
/// The source EDEM retains its exact fault laws.
/// Nonempty success outputs require a policy and are rejected.
detector_error_model dem_from_edem(const cudaq_edem_dem_view &value);
} // namespace cudaq::qec
