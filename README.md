# FFTLog-and-beyond

An extended FFTLog code for efficiently computing integrals containing:

- one Bessel function (i.e. Hankel transform)
  $$ F(y) = \int \frac{\mathrm{d}x}{x} f(x) J_{l}(xy) $$
- one spherical Bessel function or its square
  $$ F(y) = \int \frac{\mathrm{d}x}{x} f(x) j_{l}(xy) ~~\text{or}~~ F(y) = \int \frac{\mathrm{d}x}{x} f(x) j_{l}^2(xy) $$
- one 1st or 2nd-derivative of spherical Bessel function
  $$ F(y) = \int \frac{\mathrm{d}x}{x} f(x) j_{l}^{(n)}(xy) $$
- two spherical Bessel function of same/different order, same/different parameter
  $$ F(y) = \int \frac{\mathrm{d}x}{x} f(x) j_{l_1}(xy) j_{l_2}(\beta xy) $$
- two derivative of spherical Bessel function, same/different parameter
  $$ F(y) = \int \frac{\mathrm{d}x}{x} f(x) j_{l_1}'(xy) j_{l_2}'(\beta xy) $$

## Installation

The code is written in C ([./src/](src)) and provides a python wrapper ([./fftlogx/](fftlogx)). To use it, run
```shell
python setup.py install
```
to construct install the interface, then follow the test notebook provided in [/test/](test) to import and use it.

## Citation

Original work:

Please cite [Fang et al (2019); arXiv:1911.11947](https://arxiv.org/abs/1911.11947), if you find the algorithm or the code useful to your research.

Please feel free to use and adapt the code for your own purpose, and let me know if you are confused or find a bug (just open an [issue](https://github.com/xfangcosmo/FFTLog-and-beyond/issues)). FFTLog-and-beyond is open source and distributed with the
[MIT license](https://opensource.org/licenses/mit).
