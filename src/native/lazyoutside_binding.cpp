// get approximate log p and ned from the same folding pass
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include "assignments.h"

int quiet_fprintf(FILE* stream, const char* format, ...) {
    if (stream == stderr) return 0;
    va_list args;
    va_start(args, format);
    int count = std::vfprintf(stream, format, args);
    va_end(args);
    return count;
}

// reuse the upstream code without its command line or debug messages
#define main lazyoutside_cli_main
#define fprintf quiet_fprintf
#include "external/LazyOutside/src/LinearPartition.cpp"
#include "external/LazyOutside/src/LinearFoldEval.h"
#undef main
#undef fprintf

// use the upstream rt value; these energy tables are fixed at 37 c
constexpr double CENTIKCAL_PER_KCAL = 100.;
constexpr double RT = kT / CENTIKCAL_PER_KCAL;
constexpr int DANGLE_MODEL = 2;
// keep the same five-decimal energy rounding as earlier lp runs
constexpr double ENSEMBLE_ENERGY_SCALE = 100000.;
// these settings are unused because mea and threshknot are off
constexpr double UNUSED_MEA_GAMMA = 3.;
constexpr double UNUSED_THRESHKNOT_THRESHOLD = .3;

void prepare_sequence(std::string& sequence, const std::string& target) {
    if (sequence.empty() || sequence.size() != target.size())
        throw std::invalid_argument("sequence and target lengths differ");
    for (char& base : sequence) {
        base = std::toupper(static_cast<unsigned char>(base));
        if (base == 'T') base = 'U';
        if (std::string("ACGU").find(base) == std::string::npos)
            throw std::invalid_argument("invalid RNA sequence");
    }
}

std::vector<std::pair<double, double>> score_batch(
    std::vector<std::string> sequences, const std::string& target,
    int beam, double threshold, bool lazy
) {
    if (beam < 1 || !std::isfinite(threshold) || threshold <= 0 || threshold >= 1)
        throw std::invalid_argument("invalid beam or lazy threshold");
    std::vector<std::pair<double, double>> results;
    results.reserve(sequences.size());
    for (auto sequence : sequences) {
        prepare_sequence(sequence, target);
        BeamCKYParser parser(beam, true, false, "", "", false, 0., "", false, UNUSED_MEA_GAMMA,
                             "", false, false, UNUSED_THRESHKNOT_THRESHOLD, "", "", false, lazy, -std::log(threshold));
        parser.parse(sequence);
        double ned = parser.target_ned(target);
        double energy = -eval(sequence, target, false, DANGLE_MODEL) / CENTIKCAL_PER_KCAL;
        // keep the same rt and log partition values as the benchmark
        double logp = std::min(0., -energy / RT - parser.log_partition());
        results.emplace_back(logp, ned);
    }
    return results;
}

// lp and hybrid scoring only need the inside pass
std::vector<double> score_log_batch(
    std::vector<std::string> sequences, const std::string& target, int beam
) {
    if (beam < 1) throw std::invalid_argument("beam must be positive");
    std::vector<double> results;
    results.reserve(sequences.size());
    for (auto sequence : sequences) {
        prepare_sequence(sequence, target);
        BeamCKYParser parser(beam, true, false, "", "", true, 0., "", false, UNUSED_MEA_GAMMA,
                             "", false, false, UNUSED_THRESHKNOT_THRESHOLD, "", "", false, true, 0. /* outside pass disabled */);
        parser.parse(sequence);
        double energy = -eval(sequence, target, false, DANGLE_MODEL) / CENTIKCAL_PER_KCAL;
        // keep the old lp scorer's five-decimal energy rounding
        double ensemble = std::round(-RT * parser.log_partition() * ENSEMBLE_ENERGY_SCALE) / ENSEMBLE_ENERGY_SCALE;
        double logp = std::min(0., (ensemble - energy) / RT);
        results.push_back(logp);
    }
    return results;
}

PYBIND11_MODULE(_lazyoutside_native, module) {
    // use the same defaults as the python settings
    auto config = pybind11::module_::import("cem_rna").attr("Config");
    module.def("assignments_to_sequences", &assignments_to_sequences);
    module.def("score_batch", &score_batch, pybind11::arg("sequences"),
               pybind11::arg("target"), pybind11::arg("beam") = config.attr("beam"),
               pybind11::arg("threshold") = config.attr("lazy_threshold"), pybind11::arg("lazy") = true,
               pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("score_log_batch", &score_log_batch, pybind11::arg("sequences"),
               pybind11::arg("target"), pybind11::arg("beam") = config.attr("beam"),
               pybind11::call_guard<pybind11::gil_scoped_release>());
}
