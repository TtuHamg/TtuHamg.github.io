---
title: Pytorch源码理解
date: 2023-11-06 21:17:15
aside: false

---


## 代码生成 

Pytorch对函数进行debug，大部分会跳转到`torch/_C/_VariableFunctions.pyi`中。但是在Pytorch源码中仅有`torch/_C/_VariableFunctions.pyi.in`文件。在`torch/_C/_VariableFunctions.pyi`文件首行有

```python
# @generated from torch/_C/_VariableFunctions.pyi.in
```

---

pyi文件中的i指interface。

pyi implements "stub" file (definition from Martin Fowler)

Stubs: provide canned answers to calls made during the test, usually not responding at all to anything outside what's programmed in for the test.

---

Pytorch中的代码生成分为两步：

1. 由py生成pyi.in
2. 由pyi.in生成pyi

---

在Pytorch源码中由以下pyi.in文件：

- torch/\_C/\__init\__.pyi.in
- _torch/\_C/\_nn.pyi.in_
- _torch/\_C/return_types.pyi.in
- torch/\_C/_VariableFunctions.pyi.in
- torch/nn/functional.pyi.in
- torch/utils/data/datapipes/datapipe.pyi.in

这些文件实际上也是通过stubgen生成得到。在torch/CMakeLists.txt中有

```cmake
#新增文件名为torch_python_stubs的custom target，以来以下pyi文件。target和command命令搭配使用，此时用法是target依赖于command的输出（OUTPUT）
add_custom_target(torch_python_stubs DEPENDS
    "${TORCH_SRC_DIR}/_C/__init__.pyi"
    "${TORCH_SRC_DIR}/_C/_VariableFunctions.pyi"
    "${TORCH_SRC_DIR}/nn/functional.pyi"
    "${TORCH_SRC_DIR}/utils/data/datapipes/datapipe.pyi"
)

file(GLOB_RECURSE torchgen_python "${PROJECT_SOURCE_DIR}/torchgen/*.py")
file(GLOB_RECURSE autograd_python "${TOOLS_PATH}/autograd/*.py")
file(GLOB_RECURSE pyi_python "${TOOLS_PATH}/pyi/*.py")
#通过"${PYTHON_EXECUTABLE}" -mtools.pyi.gen_pyi调用tools/pyi/gen_pyi.py,输入时DEPENDS中的3个pyi.in文件，生成OUTPUT中的三个pyi文件
add_custom_command(
    OUTPUT
    "${TORCH_SRC_DIR}/_C/__init__.pyi"
    "${TORCH_SRC_DIR}/_C/_VariableFunctions.pyi"
    "${TORCH_SRC_DIR}/nn/functional.pyi"
    COMMAND
    "${PYTHON_EXECUTABLE}" -mtools.pyi.gen_pyi
      --native-functions-path "aten/src/ATen/native/native_functions.yaml"
      --tags-path "aten/src/ATen/native/tags.yaml"
      --deprecated-functions-path "tools/autograd/deprecated.yaml"
    DEPENDS
      "${TORCH_SRC_DIR}/_C/__init__.pyi.in"
      "${TORCH_SRC_DIR}/_C/_VariableFunctions.pyi.in"
      "${TORCH_SRC_DIR}/nn/functional.pyi.in"
      "${TORCH_ROOT}/aten/src/ATen/native/native_functions.yaml"
      "${TORCH_ROOT}/aten/src/ATen/native/tags.yaml"
      "${TORCH_ROOT}/tools/autograd/deprecated.yaml"
      ${pyi_python}
      ${autograd_python}
      ${torchgen_python}
    WORKING_DIRECTORY
    "${TORCH_ROOT}"
)
file(GLOB_RECURSE datapipe_files "${TORCH_SRC_DIR}/utils/data/datapipes/*.py")
add_custom_command(
    OUTPUT
    "${TORCH_SRC_DIR}/utils/data/datapipes/datapipe.pyi"
    COMMAND
    "${PYTHON_EXECUTABLE}" ${TORCH_SRC_DIR}/utils/data/datapipes/gen_pyi.py
    DEPENDS
    "${TORCH_SRC_DIR}/utils/data/datapipes/datapipe.pyi.in"
    ${datapipe_files}
    WORKING_DIRECTORY
    "${TORCH_ROOT}"
)
```

疑问：`torch/_C/_nn.pyi`和`torch/_C/return_types.pyi`也是由`tools/pyi/gen_pyi.py`生成的，为什么没有在`add_custom_target`和`add_custom_command`的`DEPENDS`和`OUTPUT`中?

之后，新增一个`torch_python`的shared library，运行后生成`build/lib/libtorch_python.so`，接着说明`torch_python`依赖于`torch_python_stubs`

```cmake
add_library(torch_python SHARED ${TORCH_PYTHON_SRCS})
add_dependencies(torch_python torch_python_stubs)
```

在非MacOS的系统上都会建构一个名為nnapi_backend的library，它的依赖中就有torch_python。

```cmake
# Skip building this library under MacOS, since it is currently failing to build on Mac
# Github issue #61930
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # Add Android Nnapi delegate library
  add_library(nnapi_backend SHARED
          ${TORCH_SRC_DIR}/csrc/jit/backends/nnapi/nnapi_backend_lib.cpp
          ${TORCH_SRC_DIR}/csrc/jit/backends/nnapi/nnapi_backend_preprocess.cpp
          )
  # Pybind11 requires explicit linking of the torch_python library
  target_link_libraries(nnapi_backend PRIVATE torch torch_python pybind::pybind11)
endif()
```

总结一下，就是有nnapi_backend -> torch_python -> torch_python_stubs -> torch/\_C/\__init\__.pyi, torch/\_C/_VariableFunctions.pyi, torch/nn/functional.pyi间的层层依赖，所以要建构nnapi_backend这个library时才会调用tools/pyi/gen_pyi.py去生成.pyi档。

---

gen_pyi.py中主函数如下：

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate type stubs for PyTorch")
    parser.add_argument(
        "--native-functions-path",
        metavar="NATIVE",
        default="aten/src/ATen/native/native_functions.yaml",
        help="path to native_functions.yaml",
    )
    parser.add_argument(
        "--tags-path",
        metavar="TAGS",
        default="aten/src/ATen/native/tags.yaml",
        help="path to tags.yaml",
    )
    parser.add_argument(
        "--deprecated-functions-path",
        metavar="DEPRECATED",
        default="tools/autograd/deprecated.yaml",
        help="path to deprecated.yaml",
    )
    parser.add_argument(
        "--out", metavar="OUT", default=".", help="path to output directory"
    )
    args = parser.parse_args()
    fm = FileManager(install_dir=args.out, template_dir=".", dry_run=False)
    gen_pyi(
        args.native_functions_path, args.tags_path, args.deprecated_functions_path, fm
    )


if __name__ == "__main__":
    main()

```

其关键在于gen_pyi函数

```python
def gen_pyi(
    native_yaml_path: str,
    tags_yaml_path: str,
    deprecated_yaml_path: str,
    fm: FileManager,
) -> None:
    """gen_pyi()

    This function generates a pyi file for torch.
    """
    
    ...
    #解析native_functions.yaml和tags.yaml，得到native_functions变量。native_functions时一个NativeFunction列别，表示aten命名空间里的函数
    native_functions = parse_native_yaml(
        native_yaml_path, tags_yaml_path
    ).native_functions
    native_functions = list(filter(should_generate_py_binding, native_functions))
```

以rand函数为例，在`native_functions.yaml`中，有以下6个override name，分别为`names`, `generator_with_names`, `空字串`, `generator`, `out`, `generator_out`。可以发现names和generator_with_names有autogen，分别有name_out和generator_with_names_out另两个out版本函数，因此native_functions.yaml里的6个rand相关函数最终对应_VariableFunctions.pyi中的8个rand函数

```yaml
- func: rand.names(SymInt[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  device_check: NoCheck
  device_guard: False
  dispatch:
    CompositeExplicitAutograd: rand
  autogen: rand.names_out
  tags: nondeterministic_seeded

- func: rand.generator_with_names(SymInt[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  device_check: NoCheck
  device_guard: False
  tags: nondeterministic_seeded
  dispatch:
    CompositeExplicitAutograd: rand
  autogen: rand.generator_with_names_out

- func: rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  tags: [core, nondeterministic_seeded]
  dispatch:
    CompositeExplicitAutograd: rand

- func: rand.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  tags: nondeterministic_seeded
  dispatch:
    CompositeExplicitAutograd: rand

- func: rand.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
  tags: nondeterministic_seeded
  dispatch:
    CompositeExplicitAutograd: rand_out

- func: rand.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
  tags: nondeterministic_seeded

```

```python
@overload
def rand(size: _size, *, generator: Optional[Generator], names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def rand(*size: _int, generator: Optional[Generator], names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def rand(size: _size, *, generator: Optional[Generator], out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def rand(*size: _int, generator: Optional[Generator], out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def rand(size: _size, *, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def rand(*size: _int, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def rand(size: _size, *, names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...
@overload
def rand(*size: _int, names: Optional[Sequence[Union[str, ellipsis, None]]], dtype: Optional[_dtype]=None, layout: Optional[_layout]=None, device: Optional[Union[_device, str, None]]=None, pin_memory: Optional[_bool]=False, requires_grad: Optional[_bool]=False) -> Tensor: ...

```

 在gen_pyi.py中，继续经过`load_signatures`和`get_py_torch_functions`函数，对于rand函数例子而言，依据有没有out被整理成两两一对，共四对：第一对是`generator_with_names`和`generator_with_names_out`第二对是`generator`和`generator_out`。第三对是`names`和`names_out`。第四对是`空字串`和`out`。分别对应`_VariableFunctions.pyi`中有generator及names参数，只有generator参数，只有name参数，沒有generator和names参数。



## Dispatch机制

### 什么是Dispatch

![](./DispatchTable.png) 


## 随机数原理

