def test_rgb_image(analyzer, rgb_image):
    assert analyzer.figure_counts(rgb_image) == (1, 1, 1)


def test_gray_image(analyzer, gray_image):
    assert analyzer.figure_counts(gray_image) == (1, 1, 1)


def test_black_rgb(analyzer, black_rgb_image):
    assert analyzer.figure_counts(black_rgb_image) == (0, 0, 0)


def test_black_gray(analyzer, black_gray_image):
    assert analyzer.figure_counts(black_gray_image) == (0, 0, 0)


def test_white_rgb(analyzer, white_rgb_image):
    assert analyzer.figure_counts(white_rgb_image) == (0, 1, 0)


def test_white_gray(analyzer, white_gray_image):
    assert analyzer.figure_counts(white_gray_image) == (0, 1, 0)
