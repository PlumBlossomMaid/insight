// include/insight/insight.h
#pragma once

// Core components
#include "insight/core/array.h"
#include "insight/core/dtype.h"
#include "insight/core/shape.h"
#include "insight/core/place.h"
#include "insight/core/slice.h"

// Operations
#include "insight/ops/elementwise.h"
#include "insight/ops/random.h"
#include "insight/ops/broadcast.h"
#include "insight/ops/reduction.h"
#include "insight/ops/creation.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/signal.h"
#include "insight/ops/fft.h"
#include "insight/ops/complex.h"
#include "insight/ops/operator.h"
#include "insight/ops/linalg.h"

// I/O utilities
#include "insight/io/csv.h"
#include "insight/io/print.h"

// Plugin registry
#include "insight/plugin/op_registry.h"
#include "insight/plugin/device_ext.h"

// Utilities
#include "insight/utils/features.h"
#include "insight/utils/promotion.h"

// Initialization
#include "insight/init.h"

namespace ins {

	// Version information
	constexpr int VERSION_MAJOR = 1;
	constexpr int VERSION_MINOR = 0;
	constexpr int VERSION_PATCH = 0;

} // namespace ins