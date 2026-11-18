from numpy import linspace

# ~~~~~~~~~~~~~~~~~~~
# ~~~ White Noise ~~~
# ~~~~~~~~~~~~~~~~~~~

def white_noise_lines(
        psd: float,
        min_frequency: float,
        max_frequency: float,
        number_of_lines: int
    ) -> tuple[list[float], list[float]]:

    """ Represents white noise with the constant one-sided PSD psd between min_frequency and max_frequency (in Hz) by number_of_lines sinusoids.
    The band is split into number_of_lines bins of equal width df, each sinusoid sits in the center of its bin. A sinusoid with the amplitude a has the variance a^2 / 2.
    It carries the noise power psd * df of its bin, s.t. the total variance equals the integral of the PSD over the band: a = sqrt(2 psd df).
    The phases of the sinusoids are random and chosen by the user of the noise model.

    Returns:
        The frequencies in Hz and the line amplitudes in the unit of sqrt(psd * Hz). """

    if psd < 0:
        raise ValueError(f"The PSD of white noise cannot be negative (given: {psd}).")
    if not 0 <= min_frequency < max_frequency:
        raise ValueError(f"The frequency band needs 0 <= min_frequency < max_frequency (given: {min_frequency} Hz to {max_frequency} Hz).")
    if number_of_lines < 1:
        raise ValueError(f"White noise needs at least one line (given: {number_of_lines}).")

    bin_width = (max_frequency - min_frequency) / number_of_lines
    frequencies = linspace(min_frequency + bin_width / 2, max_frequency - bin_width / 2, number_of_lines)
    amplitude = float((2 * psd * bin_width) ** 0.5)

    return [float(frequency) for frequency in frequencies], [amplitude] * number_of_lines


# END OF FILE
