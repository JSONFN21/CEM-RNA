// turn sampled base choices into rna sequences a little faster
#pragma once
#include <pybind11/numpy.h>
using I16Array = pybind11::array_t<int16_t, pybind11::array::c_style | pybind11::array::forcecast>;
namespace py = pybind11;
std::vector<std::string> assignments_to_sequences(
    const I16Array& assignments,
    const std::vector<std::vector<int>>& positions,
    const std::vector<std::vector<std::string>>& states,
    size_t sequence_length
)
{
    auto a = assignments.unchecked<2>();
    std::vector<std::string> result(static_cast<size_t>(a.shape(0)), std::string(sequence_length, 'A'));
    for (py::ssize_t row = 0; row < a.shape(0); ++row) {
        for (size_t variable = 0; variable < positions.size(); ++variable) {
            int state = assignments.at(row, static_cast<py::ssize_t>(variable));
            const auto& letters = states.at(variable).at(static_cast<size_t>(state));
            for (size_t k = 0; k < positions[variable].size(); ++k) {
                int position = positions[variable][k];
                result[static_cast<size_t>(row)].at(static_cast<size_t>(position)) = letters.at(k);
            }
        }
    }
    return result;
}