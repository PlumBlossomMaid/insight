// include/insight/init.h
#pragma once

namespace ins {

	/**
	 * @brief Initialize Insight library.
	 *
	 * Must be called before using any Insight functionality.
	 * This function registers all available backends and kernels.
	 */
	void init();

	/**
	 * @brief Check if Insight is initialized.
	 *
	 * @return true if initialized, false otherwise.
	 */
	bool is_initialized();

} // namespace ins