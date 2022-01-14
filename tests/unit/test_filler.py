import numpy as np


def test_rgb_image(analyzer, rgb_image):
    colors = analyzer.fill_figures(rgb_image, figures=['circle', 'triangle', 'rectangle']).reshape(-1, 3).tolist()
    assert ([255, 0, 0] in colors) and ([0, 255, 0] in colors) and ([0, 0, 255] in colors)

    colors = analyzer.fill_figures(rgb_image, figures=['circle']).reshape(-1, 3).tolist()
    assert ([255, 0, 0] in colors) and ([0, 255, 0] not in colors) and ([0, 0, 255] not in colors)

    colors = analyzer.fill_figures(rgb_image, figures=['rectangle']).reshape(-1, 3).tolist()
    assert ([255, 0, 0] not in colors) and ([0, 255, 0] in colors) and ([0, 0, 255] not in colors)

    colors = analyzer.fill_figures(rgb_image, figures=['triangle']).reshape(-1, 3).tolist()
    assert ([255, 0, 0] not in colors) and ([0, 255, 0] not in colors) and ([0, 0, 255] in colors)


def test_gray_image(analyzer, gray_image):
    colors = analyzer.fill_figures(gray_image, figures=['circle', 'triangle', 'rectangle']).reshape(-1, 3).tolist()
    assert ([255, 0, 0] in colors) and ([0, 255, 0] in colors) and ([0, 0, 255] in colors)

    colors = analyzer.fill_figures(gray_image, figures=['circle']).reshape(-1, 3).tolist()
    assert ([255, 0, 0] in colors) and ([0, 255, 0] not in colors) and ([0, 0, 255] not in colors)

    colors = analyzer.fill_figures(gray_image, figures=['rectangle']).reshape(-1, 3).tolist()
    assert ([255, 0, 0] not in colors) and ([0, 255, 0] in colors) and ([0, 0, 255] not in colors)

    colors = analyzer.fill_figures(gray_image, figures=['triangle']).reshape(-1, 3).tolist()
    assert ([255, 0, 0] not in colors) and ([0, 255, 0] not in colors) and ([0, 0, 255] in colors)


def test_black_rgb(analyzer, black_rgb_image):
    colored_image = analyzer.fill_figures(black_rgb_image)
    assert np.array_equal(colored_image, np.zeros_like(colored_image))


def test_black_gray(analyzer, black_gray_image):
    colored_image = analyzer.fill_figures(black_gray_image)
    assert np.array_equal(colored_image, np.zeros(shape=(*black_gray_image.shape, 3), dtype=np.uint8))


def test_white_rgb(analyzer, white_rgb_image):
    colored_image = analyzer.fill_figures(white_rgb_image)
    assert np.array_equal(colored_image, np.zeros_like(colored_image) + (0, 255, 0))


def test_white_gray(analyzer, white_gray_image):
    colored_image = analyzer.fill_figures(white_gray_image)
    assert np.array_equal(colored_image, np.zeros(shape=(*white_gray_image.shape, 3), dtype=np.uint8) + (0, 255, 0))
