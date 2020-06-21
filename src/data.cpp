#include <hyperpose/utility/data.hpp>
#include <hyperpose/utility/parallel_for.hpp>

namespace hyperpose {

void nhwc_images_append_nchw_batch(std::vector<float>& data, std::vector<cv::Mat>& images, double factor, bool flip_rb)
{
    if (images.empty())
        return;

    const auto size = images.at(0).size();
    data.resize(size.area() * 3 * images.size() + data.size());

    hyperpose::parallel_for<size_t>(images.size(), [=, &data, &images](const size_t imageIdx)
    {
        auto&& image = images[imageIdx];
        assert(image.type() == CV_8UC3);
        assert(size == image.size());

        const bool isContinuous = image.isContinuous();
        const int iter_rows = isContinuous ? 1 : image.rows;
        const int iter_cols = isContinuous ? image.total() : image.cols;

        const size_t img_offset = (size_t(3) * iter_rows * iter_cols) * imageIdx;

        constexpr const std::array<size_t, 3> no_swap{ 0, 1, 2 };
        constexpr const std::array<size_t, 3> swap_rb{ 2, 1, 0 };
        const auto& index_ref = flip_rb ? swap_rb : no_swap;
        size_t didx = 0;
        for (size_t c : index_ref)
        {
            for (int i = 0; i < iter_rows; ++i) {
                const auto* const line = image.ptr<cv::Vec3b>(i);
                for (int j = 0; j < iter_cols; ++j)
                {
                    data[img_offset + didx++] = line[j][c] * factor;
                }
            }
        }
    });
}

} // namespace hyperpose