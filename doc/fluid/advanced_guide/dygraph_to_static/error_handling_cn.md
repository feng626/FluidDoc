# 报错信息处理
> 报错的话，介绍动转静报错，报错信息的构成，以及分类介绍 编译期报错、运行期报错、动转静过程中报错，给出示例。

本节内容将介绍使用动态图转静态图（下文简称动转静）功能发生异常时，[ProgramTranslator](./program_translator_cn.html)<!--【增加一个ProgramTranslator的链接，或者改成to_static更合适一点？】-->对报错信息做的处理，以帮助用户更好地理解动转静报错信息。使用动转静功能运行动态图代码时，内部可以分为2个步骤：动态图代码转化成静态图代码，运行静态图代码。接下来分别介绍这2个步骤的报错信息。
~~动转静后的代码运行时的报错，以及动态图代码转化到静态图代码过程中的报错。~~

## 动转静代码转换过程中报错


## 运行转化后的代码报错
如果在动转静后的静态图代码中发生异常，ProgramTranslator会捕获该异常，增强异常报错信息，将静态图代码报错行映射到转化前的动态图代码，并重新抛出该异常。以便将报错定位到原始代码中。
重新抛出的异常具有以下几个特点：

- 怎么描述这个，隐藏了信息栈
- 转化前的代码前会给出提示："In User Code:"
- 报错信息中包含了转化前的原始动态图代码


例如，运行下列代码将引发异常，在静态图构建时，即编译期<!--【给个链接】-->
```Python
import paddle
import numpy as np

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    x = paddle.reshape(x, shape=[-1, -1])
    return x

paddle.disable_static()
func(np.ones([3, 2]))
```

```Shell
Traceback (most recent call last):
<ipython-input-13-f9c3ea702e3a> in <module>()
     func(np.ones([3, 2]))
  File "paddle/fluid/dygraph/dygraph_to_static/program_translator.py", line 332, in __call__
    raise new_exception
AssertionError: In user code:

    File "<ipython-input-13-f9c3ea702e3a>", line 7, in func
        x = fluid.layers.reshape(x, shape=[-1, -1])
    File "paddle/fluid/layers/nn.py", line 6193, in reshape
        attrs["shape"] = get_attr_shape(shape)
    File "paddle/fluid/layers/nn.py", line 6169, in get_attr_shape
        "be -1. But received shape[%d] is also -1." % dim_idx)
    AssertionError: Only one dimension value of 'shape' in reshape can be -1. But received shape[1] is also -1.
```
上述报错信息可以分为3部分来看：

注意：报错栈中，涉及代码转化过程的信息栈默认会被隐藏，不对用户展示，以避免给用户带来困扰。
```
# 被隐藏的报错栈
```

ProgramTranslator处理后的报错信息中，会包含提示"In user code:"，表示之后的报错栈中，包含动转静前的动态图代码，即用户写的代码。
```
AssertionError: In user code:

    File "<ipython-input-13-f9c3ea702e3a>", line 7, in func
        x = fluid.layers.reshape(x, shape=[-1, -1])
    File "paddle/fluid/layers/nn.py", line 6193, in reshape
        attrs["shape"] = get_attr_shape(shape)
    File "paddle/fluid/layers/nn.py", line 6169, in get_attr_shape
        "be -1. But received shape[%d] is also -1." % dim_idx)
```

最后，是原始异常中的报错信息：
```
	AssertionError: Only one dimension value of 'shape' in reshape can be -1. But received shape[1] is also -1.
```


以上报错，是组网时的报错，接下来将展示运行时的报错：
```
Traceback (most recent call last):
  <ipython-input-17-d5ca39eb630e> in <module>()
    self.func(self.input)
  File "paddle/fluid/dygraph/dygraph_to_static/program_translator.py", line 332, in __call__
    raise new_exception
EnforceNotMet: In user code:

    File "<ipython-input-17-d5ca39eb630e>", line 5, in func_error_in_runtime
      x = fluid.layers.reshape(x, shape=[1, two])
    File "/home/liyamei01/anaconda2/lib/python2.7/site-packages/paddle/fluid/layers/nn.py", line 6209, in reshape
      "XShape": x_shape})
    File "/home/liyamei01/anaconda2/lib/python2.7/site-packages/paddle/fluid/layer_helper.py", line 43, in append_op
      return self.main_program.current_block().append_op(*args, **kwargs)
    File "/home/liyamei01/anaconda2/lib/python2.7/site-packages/paddle/fluid/framework.py", line 2880, in append_op
      attrs=kwargs.get("attrs", None))
    File "/home/liyamei01/anaconda2/lib/python2.7/site-packages/paddle/fluid/framework.py", line 1977, in __init__
      for frame in traceback.extract_stack():

--------------------------------------
C++ Traceback (most recent call last):
--------------------------------------
0   paddle::imperative::Tracer::TraceOp(std::string const&, paddle::imperative::NameVarBaseMap const&, paddle::imperative::NameVarBaseMap const&, paddle::framework::AttributeMap, paddle::platform::Place const&, bool)
1   paddle::imperative::OpBase::Run(paddle::framework::OperatorBase const&, paddle::imperative::NameVarBaseMap const&, paddle::imperative::NameVarBaseMap const&, paddle::framework::AttributeMap const&, paddle::platform::Place const&)
2   paddle::imperative::PreparedOp::Run(paddle::imperative::NameVarBaseMap const&, paddle::imperative::NameVarBaseMap const&, paddle::framework::AttributeMap const&)
3   std::_Function_handler<void (paddle::framework::ExecutionContext const&), paddle::framework::OpKernelRegistrarFunctor<paddle::platform::CPUPlace, false, 0ul, paddle::operators::RunProgramOpKernel<paddle::platform::CPUDeviceContext, float> >::operator()(char const*, char const*, int) const::{lambda(paddle::framework::ExecutionContext const&)#1}>::_M_invoke(std::_Any_data const&, paddle::framework::ExecutionContext const&)
4   paddle::operators::RunProgramOpKernel<paddle::platform::CPUDeviceContext, float>::Compute(paddle::framework::ExecutionContext const&) const
5   paddle::framework::Executor::RunPartialPreparedContext(paddle::framework::ExecutorPrepareContext*, paddle::framework::Scope*, long, long, bool, bool, bool)
6   paddle::framework::OperatorBase::Run(paddle::framework::Scope const&, paddle::platform::Place const&)
7   paddle::framework::OperatorWithKernel::RunImpl(paddle::framework::Scope const&, paddle::platform::Place const&) const
8   paddle::framework::OperatorWithKernel::RunImpl(paddle::framework::Scope const&, paddle::platform::Place const&, paddle::framework::RuntimeContext*) const
9   paddle::operators::ReshapeKernel::operator()(paddle::framework::ExecutionContext const&) const
10  paddle::operators::ReshapeOp::ValidateShape(std::vector<int, std::allocator<int> >, paddle::framework::DDim const&)
11  paddle::platform::EnforceNotMet::EnforceNotMet(std::string const&, char const*, int)
12  paddle::platform::GetCurrentTraceBackString()

----------------------
Error Message Summary:
----------------------
InvalidArgumentError: The 'shape' in ReshapeOp is invalid. The input tensor X'size must be equal to the capacity of 'shape'. But received X's shape = [3, 2], X's size = 6, 'shape' is [1, 2], the capacity of 'shape' is 2.
  [Hint: Expected capacity == in_size, but received capacity:2 != in_size:6.] (at /paddle/paddle/fluid/operators/reshape_op.cc:206)
  [operator < reshape2 > error]  [operator < run_program > error]
```
