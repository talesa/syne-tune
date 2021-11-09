# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# This file has been taken from Ray. The reason for reusing the file is to be able to support the same API when
# defining search space while avoiding to have Ray as a required dependency. We may want to add functionality in the
# future.
import logging
from copy import copy
from inspect import signature
from math import isclose
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import argparse

import numpy as np

logger = logging.getLogger(__name__)


class Domain:
    """Base class to specify a type and valid range to sample parameters from.

    This base class is implemented by parameter spaces, like float ranges
    (``Float``), integer ranges (``Integer``), or categorical variables
    (``Categorical``). The ``Domain`` object contains information about
    valid values (e.g. minimum and maximum values), and exposes methods that
    allow specification of specific samplers (e.g. ``uniform()`` or
    ``loguniform()``).

    """
    sampler = None
    default_sampler_cls = None

    @property
    def value_type(self):
        raise NotImplementedError

    def cast(self, value):
        """Cast value to domain type"""
        return self.value_type(value)

    def set_sampler(self, sampler, allow_override=False):
        if self.sampler and not allow_override:
            raise ValueError("You can only choose one sampler for parameter "
                             "domains. Existing sampler for parameter {}: "
                             "{}. Tried to add {}".format(
                                 self.__class__.__name__, self.sampler,
                                 sampler))
        self.sampler = sampler

    def get_sampler(self) -> "Sampler":
        sampler = self.sampler
        if not sampler:
            sampler = self.default_sampler_cls()
        return sampler

    def sample(self, spec=None, size=1, random_state=None):
        sampler = self.get_sampler()
        return sampler.sample(
            self, spec=spec, size=size, random_state=random_state)

    def is_grid(self):
        return isinstance(self.sampler, Grid)

    def is_function(self):
        return False

    def is_valid(self, value: Any):
        """Returns True if `value` is a valid value in this domain."""
        raise NotImplementedError

    @property
    def domain_str(self):
        return "(unknown)"

    def __len__(self):
        """
        :return: Size of domain (number of distinct elements), or 0 if size
            is infinite
        """
        raise NotImplementedError


class Sampler:
    def sample(self,
               domain: Domain,
               spec: Optional[Union[List[Dict], Dict]] = None,
               size: int = 1,
               random_state: Optional[np.random.RandomState] = None):
        raise NotImplementedError

class BaseSampler(Sampler):
    def __str__(self):
        return "Base"


class Uniform(Sampler):
    def __str__(self):
        return "Uniform"


class LogUniform(Sampler):
    def __init__(self, base: float = 10):
        self.base = base
        assert self.base > 0, "Base has to be strictly greater than 0"

    def __str__(self):
        return "LogUniform"


class Normal(Sampler):
    def __init__(self, mean: float = 0., sd: float = 0.):
        self.mean = mean
        self.sd = sd

        assert self.sd > 0, "SD has to be strictly greater than 0"

    def __str__(self):
        return "Normal"


class Grid(Sampler):
    """Dummy sampler used for grid search"""

    def sample(self,
               domain: Domain,
               spec: Optional[Union[List[Dict], Dict]] = None,
               size: int = 1,
               random_state: Optional[np.random.RandomState] = None):
        return RuntimeError("Do not call `sample()` on grid.")


class Float(Domain):
    class _Uniform(Uniform):
        def sample(self,
                   domain: "Float",
                   spec: Optional[Union[List[Dict], Dict]] = None,
                   size: int = 1,
                   random_state: Optional[np.random.RandomState] = None):
            assert domain.lower > float("-inf"), \
                "Uniform needs a lower bound"
            assert domain.upper < float("inf"), \
                "Uniform needs a upper bound"
            if random_state is None:
                random_state = np.random
            items = random_state.uniform(domain.lower, domain.upper, size=size)
            return items if len(items) > 1 else domain.cast(items[0])

    class _LogUniform(LogUniform):
        def sample(self,
                   domain: "Float",
                   spec: Optional[Union[List[Dict], Dict]] = None,
                   size: int = 1,
                   random_state: Optional[np.random.RandomState] = None):
            assert domain.lower > 0, \
                "LogUniform needs a lower bound greater than 0"
            assert 0 < domain.upper < float("inf"), \
                "LogUniform needs a upper bound greater than 0"
            logmin = np.log(domain.lower) / np.log(self.base)
            logmax = np.log(domain.upper) / np.log(self.base)
            if random_state is None:
                random_state = np.random
            items = self.base**(random_state.uniform(
                logmin, logmax, size=size))
            return items if len(items) > 1 else domain.cast(items[0])

    class _Normal(Normal):
        def sample(self,
                   domain: "Float",
                   spec: Optional[Union[List[Dict], Dict]] = None,
                   size: int = 1,
                   random_state: Optional[np.random.RandomState] = None):
            assert not domain.lower or domain.lower == float("-inf"), \
                "Normal sampling does not allow a lower value bound."
            assert not domain.upper or domain.upper == float("inf"), \
                "Normal sampling does not allow a upper value bound."
            if random_state is None:
                random_state = np.random
            items = random_state.normal(self.mean, self.sd, size=size)
            return items if len(items) > 1 else domain.cast(items[0])

    default_sampler_cls = _Uniform

    def __init__(self, lower: Optional[float], upper: Optional[float]):
        # Need to explicitly check for None
        self.lower = lower if lower is not None else float("-inf")
        self.upper = upper if upper is not None else float("inf")

    @property
    def value_type(self):
        return float

    def uniform(self):
        if not self.lower > float("-inf"):
            raise ValueError(
                "Uniform requires a lower bound. Make sure to set the "
                "`lower` parameter of `Float()`.")
        if not self.upper < float("inf"):
            raise ValueError(
                "Uniform requires a upper bound. Make sure to set the "
                "`upper` parameter of `Float()`.")
        new = copy(self)
        new.set_sampler(self._Uniform())
        return new

    def loguniform(self, base: float = 10):
        if not self.lower > 0:
            raise ValueError(
                "LogUniform requires a lower bound greater than 0."
                f"Got: {self.lower}. Did you pass a variable that has "
                "been log-transformed? If so, pass the non-transformed value "
                "instead.")
        if not 0 < self.upper < float("inf"):
            raise ValueError(
                "LogUniform requires a upper bound greater than 0. "
                f"Got: {self.lower}. Did you pass a variable that has "
                "been log-transformed? If so, pass the non-transformed value "
                "instead.")
        new = copy(self)
        new.set_sampler(self._LogUniform(base))
        return new

    def normal(self, mean=0., sd=1.):
        new = copy(self)
        new.set_sampler(self._Normal(mean, sd))
        return new

    def quantized(self, q: float):
        if self.lower > float("-inf") and not isclose(self.lower / q,
                                                      round(self.lower / q)):
            raise ValueError(
                f"Your lower variable bound {self.lower} is not divisible by "
                f"quantization factor {q}.")
        if self.upper < float("inf") and not isclose(self.upper / q,
                                                     round(self.upper / q)):
            raise ValueError(
                f"Your upper variable bound {self.upper} is not divisible by "
                f"quantization factor {q}.")

        new = copy(self)
        new.set_sampler(Quantized(new.get_sampler(), q), allow_override=True)
        return new

    def is_valid(self, value: float):
        return self.lower <= value <= self.upper

    @property
    def domain_str(self):
        return f"({self.lower}, {self.upper})"

    def __len__(self):
        if self.lower < self.upper:
            return 0
        else:
            return 1


class Integer(Domain):
    class _Uniform(Uniform):
        def sample(self,
                   domain: "Integer",
                   spec: Optional[Union[List[Dict], Dict]] = None,
                   size: int = 1,
                   random_state: Optional[np.random.RandomState] = None):
            if random_state is None:
                random_state = np.random
            # Note: domain.upper is inclusive here, but exclusive in
            # `np.random.randint`.
            items = random_state.randint(
                domain.lower, domain.upper + 1, size=size)
            return items if len(items) > 1 else domain.cast(items[0])

    class _LogUniform(LogUniform):
        def sample(self,
                   domain: "Integer",
                   spec: Optional[Union[List[Dict], Dict]] = None,
                   size: int = 1,
               random_state: Optional[np.random.RandomState] = None):
            assert domain.lower > 0, \
                "LogUniform needs a lower bound greater than 0"
            assert 0 < domain.upper < float("inf"), \
                "LogUniform needs a upper bound greater than 0"
            logmin = np.log(domain.lower) / np.log(self.base)
            logmax = np.log(domain.upper) / np.log(self.base)
            if random_state is None:
                random_state = np.random
            items = self.base**(random_state.uniform(
                logmin, logmax, size=size))
            items = np.round(items).astype(int)
            return items if len(items) > 1 else domain.cast(items[0])

    default_sampler_cls = _Uniform

    def __init__(self, lower, upper):
        self.lower = self.cast(lower)
        self.upper = self.cast(upper)

    @property
    def value_type(self):
        return int

    def quantized(self, q: int):
        new = copy(self)
        new.set_sampler(Quantized(new.get_sampler(), q), allow_override=True)
        return new

    def uniform(self):
        new = copy(self)
        new.set_sampler(self._Uniform())
        return new

    def loguniform(self, base: float = 10):
        if not self.lower > 0:
            raise ValueError(
                "LogUniform requires a lower bound greater than 0."
                f"Got: {self.lower}. Did you pass a variable that has "
                "been log-transformed? If so, pass the non-transformed value "
                "instead.")
        if not 0 < self.upper < float("inf"):
            raise ValueError(
                "LogUniform requires a upper bound greater than 0. "
                f"Got: {self.lower}. Did you pass a variable that has "
                "been log-transformed? If so, pass the non-transformed value "
                "instead.")
        new = copy(self)
        new.set_sampler(self._LogUniform(base))
        return new

    def is_valid(self, value: int):
        return self.lower <= value <= self.upper

    @property
    def domain_str(self):
        return f"({self.lower}, {self.upper})"

    def __len__(self):
        return self.upper - self.lower + 1


class Categorical(Domain):
    class _Uniform(Uniform):
        def sample(self,
                   domain: "Categorical",
                   spec: Optional[Union[List[Dict], Dict]] = None,
                   size: int = 1,
                   random_state: Optional[np.random.RandomState] = None):
            if random_state is None:
                random_state = np.random
            categories = domain.categories
            items = [categories[i] for i in random_state.choice(
                len(categories), size=size)]
            return items if len(items) > 1 else domain.cast(items[0])

    default_sampler_cls = _Uniform

    def __init__(self, categories: Sequence):
        assert len(categories) > 0
        self.categories = list(categories)
        if not isinstance(self.value_type, int) and not isinstance(self.value_type, str):
            logger.info("Categorical value will be converted to string to avoid float conversion and "
                        "serialization issues.")

    def uniform(self):
        new = copy(self)
        new.set_sampler(self._Uniform())
        return new

    def grid(self):
        new = copy(self)
        new.set_sampler(Grid())
        return new

    def __len__(self):
        return len(self.categories)

    def __getitem__(self, item):
        return self.categories[item]

    def is_valid(self, value: Any):
        return value in self.categories

    @property
    def value_type(self):
        return type(self.categories[0])
        if isinstance(self.categories[0], int):
            return int
        else:
            return str

    @property
    def domain_str(self):
        return f"{self.categories}"


class Function(Domain):
    class _CallSampler(BaseSampler):
        def sample(self,
                   domain: "Function",
                   spec: Optional[Union[List[Dict], Dict]] = None,
                   size: int = 1,
                   random_state: Optional[np.random.RandomState] = None):
            if random_state is not None:
                raise NotImplementedError()
            if domain.pass_spec:
                items = [
                    domain.func(spec[i] if isinstance(spec, list) else spec)
                    for i in range(size)
                ]
            else:
                items = [domain.func() for i in range(size)]

            return items if len(items) > 1 else domain.cast(items[0])

    default_sampler_cls = _CallSampler

    def __init__(self, func: Callable):
        sig = signature(func)

        pass_spec = True  # whether we should pass `spec` when calling `func`
        try:
            sig.bind({})
        except TypeError:
            pass_spec = False

        if not pass_spec:
            try:
                sig.bind()
            except TypeError as exc:
                raise ValueError(
                    "The function passed to a `Function` parameter must be "
                    "callable with either 0 or 1 parameters.") from exc

        self.pass_spec = pass_spec
        self.func = func

    def is_function(self):
        return True

    def is_valid(self, value: Any):
        return True  # This is user-defined, so lets not assume anything

    @property
    def domain_str(self):
        return f"{self.func}()"

    def __len__(self):
        return 0


class Quantized(Sampler):
    def __init__(self, sampler: Sampler, q: Union[float, int]):
        self.sampler = sampler
        self.q = q

        assert self.sampler, "Quantized() expects a sampler instance"

    def get_sampler(self):
        return self.sampler

    def sample(self,
               domain: Domain,
               spec: Optional[Union[List[Dict], Dict]] = None,
               size: int = 1,
               random_state: Optional[np.random.RandomState] = None):
        values = self.sampler.sample(domain, spec, size, random_state)
        quantized = np.round(np.divide(values, self.q)) * self.q
        if not isinstance(quantized, np.ndarray):
            return domain.cast(quantized)
        return list(quantized)


def sample_from(func: Callable[[Dict], Any]):
    """Specify that tune should sample configuration values from this function.

    Arguments:
        func: An callable function to draw a sample from.
    """
    return Function(func)


def uniform(lower: float, upper: float):
    """Sample a float value uniformly between ``lower`` and ``upper``.

    Sampling from ``tune.uniform(1, 10)`` is equivalent to sampling from
    ``np.random.uniform(1, 10))``

    """
    return Float(lower, upper).uniform()


def quniform(lower: float, upper: float, q: float):
    """Sample a quantized float value uniformly between ``lower`` and ``upper``.

    Sampling from ``tune.uniform(1, 10)`` is equivalent to sampling from
    ``np.random.uniform(1, 10))``

    The value will be quantized, i.e. rounded to an integer increment of ``q``.
    Quantization makes the upper bound inclusive.

    """
    return Float(lower, upper).uniform().quantized(q)


def loguniform(lower: float, upper: float, base: float = 10):
    """Sugar for sampling in different orders of magnitude.

    Args:
        lower (float): Lower boundary of the output interval (e.g. 1e-4)
        upper (float): Upper boundary of the output interval (e.g. 1e-2)
        base (int): Base of the log. Defaults to 10.

    """
    return Float(lower, upper).loguniform(base)


def qloguniform(lower: float, upper: float, q: float, base: float = 10):
    """Sugar for sampling in different orders of magnitude.

    The value will be quantized, i.e. rounded to an integer increment of ``q``.

    Quantization makes the upper bound inclusive.

    Args:
        lower (float): Lower boundary of the output interval (e.g. 1e-4)
        upper (float): Upper boundary of the output interval (e.g. 1e-2)
        q (float): Quantization number. The result will be rounded to an
            integer increment of this value.
        base (int): Base of the log. Defaults to 10.

    """
    return Float(lower, upper).loguniform(base).quantized(q)


def choice(categories: List):
    """Sample a categorical value.

    Sampling from ``tune.choice([1, 2])`` is equivalent to sampling from
    ``random.choice([1, 2])``

    """
    return Categorical(categories).uniform()


def randint(lower: int, upper: int):
    """Sample an integer value uniformly between ``lower`` and ``upper``.

    ``lower`` and ``upper`` are inclusive. This is a difference to Ray Tune,
    where ``upper`` is exclusive. However, both `lograndint` and `qrandint`
    have inclusive ``upper`` in Ray Tune, so we fix this inconsistency here.

    Sampling from ``tune.randint(10)`` is equivalent to sampling from
    ``np.random.randint(10 + 1)``.

    """
    return Integer(lower, upper).uniform()


def lograndint(lower: int, upper: int, base: float = 10):
    """Sample an integer value log-uniformly between ``lower`` and ``upper``,
    with ``base`` being the base of logarithm.

    ``lower`` and ``upper` are inclusive.

    """
    return Integer(lower, upper).loguniform(base)


def qrandint(lower: int, upper: int, q: int = 1):
    """Sample an integer value uniformly between ``lower`` and ``upper``.

    ``lower`` is inclusive, ``upper`` is also inclusive (!).

    The value will be quantized, i.e. rounded to an integer increment of ``q``.
    Quantization makes the upper bound inclusive.

    """
    return Integer(lower, upper).uniform().quantized(q)


def qlograndint(lower: int, upper: int, q: int, base: float = 10):
    """Sample an integer value log-uniformly between ``lower`` and ``upper``,
    with ``base`` being the base of logarithm.

    ``lower`` is inclusive, ``upper`` is also inclusive (!).

    The value will be quantized, i.e. rounded to an integer increment of ``q``.
    Quantization makes the upper bound inclusive.

    """
    return Integer(lower, upper).loguniform(base).quantized(q)


def randn(mean: float = 0., sd: float = 1.):
    """Sample a float value normally with ``mean`` and ``sd``.

    Args:
        mean (float): Mean of the normal distribution. Defaults to 0.
        sd (float): SD of the normal distribution. Defaults to 1.

    """
    return Float(None, None).normal(mean, sd)


def qrandn(mean: float, sd: float, q: float):
    """Sample a float value normally with ``mean`` and ``sd``.

    The value will be quantized, i.e. rounded to an integer increment of ``q``.

    Args:
        mean (float): Mean of the normal distribution.
        sd (float): SD of the normal distribution.
        q (float): Quantization number. The result will be rounded to an
            integer increment of this value.

    """
    return Float(None, None).normal(mean, sd).quantized(q)


def is_log_space(domain: Domain) -> bool:
    sampler = domain.get_sampler()
    return isinstance(sampler, Float._LogUniform) or isinstance(sampler, Integer._LogUniform)


def add_to_argparse(parser: argparse.ArgumentParser, config_space: Dict):
    """
    Use this to prepare argument parser in endpoint script, for the
    non-fixed parameters in `config_space`.

    :param parser:
    :param config_space:
    :return:
    """
    for name, domain in config_space.items():
        tp = domain.value_type if isinstance(domain, Domain) else type(domain)
        parser.add_argument(f"--{name}", type=tp, required=True)


def cast_config_values(config: Dict, config_space: Dict) -> Dict:
    """
    Returns config with keys, values of `config`, but values are casted to
    their specific types.

    :param config: Config whose values are to be casted
    :param config_space:
    :return: New config with values casted to correct types
    """
    return {
        name: domain.cast(config[name]) if isinstance(domain, Domain) else config[name]
        for name, domain in config_space.items()
        if name in config
    }


def non_constant_hyperparameter_keys(config_space: Dict) -> List[str]:
    """
    :param config_space:
    :return: Keys corresponding to (non-fixed) hyperparameters
    """
    return [name for name, domain in config_space.items()
            if isinstance(domain, Domain)]


def search_space_size(config_space: Dict, upper_limit: int = 2 ** 20) -> Optional[int]:
    """
    Counts the number of distinct configurations in the search space
    `config_space`. If this is infinite (due to real-valued parameters) or
    larger than `upper_limit`, None is returned.
    """
    assert upper_limit > 1
    size = 1
    for name, domain in config_space.items():
        if isinstance(domain, Domain):
            domain_size = len(domain)
            if domain_size == 0 or domain_size > upper_limit:
                return None  # Try to avoid overflow
            size *= domain_size
            if size > upper_limit:
                return None
    return size


def to_dict(x: "Domain") -> Dict:
    domain_kwargs = {k: v for k, v in x.__dict__.items() if k != 'sampler'}
    return {
        "domain_cls": x.__class__.__name__,
        "domain_kwargs": domain_kwargs,
        "sampler_cls": str(x.sampler),
        "sampler_kwargs": x.get_sampler().__dict__
    }


def from_dict(d: Dict) -> Domain:
    domain_cls = getattr(sys.modules[__name__], d["domain_cls"])
    domain_kwargs = d["domain_kwargs"]
    sampler_cls = getattr(domain_cls, "_" + d["sampler_cls"])
    sampler_kwargs = d["sampler_kwargs"]

    if "cast_str" in domain_kwargs:
        domain_kwargs.pop("cast_str")

    domain = domain_cls(**domain_kwargs)
    sampler = sampler_cls(**sampler_kwargs)
    domain.set_sampler(sampler)
    return domain