from typing import Protocol, Self

from numpy import ndarray
from torch import Tensor


class SimpleClassProtocol(Protocol):
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __getattr__(self, attr): ...


class BaseTensorProtocol(SimpleClassProtocol, Protocol):

    def __getitem__(self, idx) -> Self: ...
    def __len__(self) -> int: ...

    def to(self, *args, **kwargs) -> Self: ...
    def cuda(self) -> Self: ...
    def numpy(self) -> Self: ...
    def cpu(self) -> Self: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    data: Tensor


class BoxesProtocol(BaseTensorProtocol, Protocol):
    def __init__(
        self, boxes: Tensor | ndarray, orig_shape: tuple[int, int]
    ) -> None: ...

    def update(self, *args, **kwargs) -> None: ...

    @property
    def id(self) -> Tensor | None: ...
    @property
    def xyxy(self) -> Tensor: ...
    @property
    def conf(self) -> Tensor: ...
    @property
    def cls(self) -> Tensor: ...

    @property
    def is_base_box(self) -> Tensor: ...
    @property
    def base_box_id(self) -> Tensor: ...


class MasksProtocol(BaseTensorProtocol, Protocol):
    @property
    def xy(self) -> list[Tensor]: ...


class OcrsProtocol(BaseTensorProtocol, Protocol):
    def update(self, *args, **kwargs): ...

    @property
    def xyxy(self) -> Tensor: ...
    @property
    def index(self) -> Tensor: ...
    @property
    def conf(self) -> Tensor: ...
    @property
    def number(self) -> Tensor: ...


class AggregateProtocol(BaseTensorProtocol, Protocol):

    @property
    def frame_no(self) -> Tensor:
        return self.data[:, 0]

    @property
    def is_base_box(self) -> Tensor:
        return self.data[:, 1]

    @property
    def base_box_id(self) -> Tensor:
        return self.data[:, 2]

    @property
    def id(self) -> Tensor:
        return self.data[:, 3]

    @property
    def box_conf(self) -> Tensor:
        return self.data[:, 4]

    @property
    def cls(self) -> Tensor:
        return self.data[:, 5]

    @property
    def conf(self) -> Tensor:
        return self.data[:, 6]

    @property
    def number(self) -> Tensor:
        return self.data[:, 7]


class ResultsProtocol(SimpleClassProtocol, Protocol):

    def __getitem__(self, idx) -> Self: ...
    def __len__(self) -> int | None: ...

    def cuda(self) -> Self: ...
    def numpy(self) -> Self: ...
    def cpu(self) -> Self: ...

    def update(self, *args, **kwargs) -> None: ...
    def to(self, *args, **kwargs) -> Self: ...
    def plot(self, *args, **kwargs): ...

    boxes: BoxesProtocol
    masks: MasksProtocol
    ocrs: OcrsProtocol | None
    aggregate: AggregateProtocol

    orig_img: ndarray
    orig_shape: tuple[int, int]

    names: dict[int, str]
    path: str
    frame_no: int
