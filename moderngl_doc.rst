================================================================================
Combined Documentation
================================================================================


.. File: index.rst

ModernGL
========

ModernGL is a high performance rendering module for Python.

.. toctree::
    :maxdepth: 2

    install/index.rst
    the_guide/index.rst
    topics/index.rst
    techniques/index.rst
    reference/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


--------------------------------------------------------------------------------


.. File: install/index.rst


Install
=======

.. toctree::
    :maxdepth: 1

    installation.rst
    using-moderngl-in-ci.rst


--------------------------------------------------------------------------------


.. File: the_guide/index.rst

The Guide
=========

Consistent learning of ModernGL.

.. toctree::
    :maxdepth: 2

    intro.rst
    getting_started/index.rst
    functionality_expansion/index.rst


--------------------------------------------------------------------------------


.. File: the_guide/functionality_expansion/index.rst

Functionality expansion
=======================

Opening up new possibilities with ModernGL.

.. toctree::
    :maxdepth: 2

    window_using.rst


--------------------------------------------------------------------------------


.. File: the_guide/getting_started/index.rst

Getting started with ModernGL
=============================

A comprehensive guide to getting you started with ModernGL.
No experience with OpenGL is required, just an understanding of how it works.

.. toctree::
    :maxdepth: 2

    low_start/index.rst
    first_rendering/index.rst
    triangles_draw/index.rst


--------------------------------------------------------------------------------


.. File: the_guide/getting_started/triangles_draw/index.rst

Triangles drawing
=================

Drawing the first triangles. Closer to real rendering.

.. toctree::
    :maxdepth: 2

    one_familiar_triangle.rst
    uniform.rst


--------------------------------------------------------------------------------


.. File: the_guide/getting_started/low_start/index.rst

Low start
=========

Low start in learning ModernGL.
Coverage of all ModernGL objects with a glance and first use of GPU.

.. toctree::
    :maxdepth: 2

    creating_context.rst
    moderngl_types.rst
    shader_basics.rst
    shader_transform.rst


--------------------------------------------------------------------------------


.. File: the_guide/getting_started/first_rendering/index.rst

First rendering
===============

Unlike OpenGL, ModernGL requires far fewer lines due to robust function generalization, which also reduces the likelihood of bugs when porting code to different platforms.

In this example, we will draw a broken line, which will look different each time we run the code, and at the same time we will slightly expand our horizons in understanding the already mentioned ModernGL objects.

.. toctree::
    :maxdepth: 2

    program.rst
    buffer.rst
    vertex_array.rst
    rendering.rst


--------------------------------------------------------------------------------


.. File: techniques/index.rst

Techniques
==========

.. toctree::
    :maxdepth: 2

    headless_ubuntu_18_server.rst


--------------------------------------------------------------------------------


.. File: reference/index.rst

Reference
=========

.. toctree::
    :maxdepth: 1

    moderngl.rst
    context.rst
    buffer.rst
    vertex_array.rst
    program.rst
    sampler.rst
    texture.rst
    texture_array.rst
    texture3d.rst
    texture_cube.rst
    framebuffer.rst
    renderbuffer.rst
    scope.rst
    query.rst
    compute_shader.rst


--------------------------------------------------------------------------------


.. File: topics/index.rst


Topics
======

.. toctree::
    :maxdepth: 2

    gc.rst
    context.rst
    texture_formats.rst
    buffer_format.rst


--------------------------------------------------------------------------------


.. File: install/installation.rst


From PyPI (pip)
---------------

ModernGL is available on PyPI for Windows, OS X and Linux as pre-built
wheels. No complication is needed unless you are setting up a
development environment.

.. code-block:: sh

    $ pip install moderngl

Verify that the package is working:

.. code:: sh

    $ python -m moderngl
    moderngl 5.6.0
    --------------
    vendor: NVIDIA Corporation
    renderer: GeForce RTX 2080 SUPER/PCIe/SSE2
    version: 3.3.0 NVIDIA 441.87
    python: 3.7.6 (tags/v3.7.6:43364a7ae0, Dec 19 2019, 00:42:30) [MSC v.1916 64 bit (AMD64)]
    platform: win32
    code: 330

.. Note:: If you experience issues it's probably related to context creation.
          More configuration might be needed to run moderngl in some cases.
          This is especially true on linux running without X. See the context section.

Development Environment
-----------------------

Ideally you want to fork the repository first.

.. code-block:: sh

    # .. or clone for your fork
    git clone https://github.com/moderngl/moderngl.git
    cd moderngl

Building on various platforms:

* On Windows you need visual c++ build tools installed:
  https://visualstudio.microsoft.com/visual-cpp-build-tools/
* On OS X you need X Code installed + command line tools
  (``xcode-select --install``)
* Building on linux should pretty much work out of the box
* To compile moderngl: ``python setup.py build_ext --inplace``

Package and dev dependencies:

* Install ``requirements.txt``, ``tests/requirements.txt`` and ``docs/requirements.txt``
* Install the package in editable mode: ``pip install -e .``

Using with Mesa 3D on Windows
-----------------------------

If you have an old Graphics Card that raises errors when running moderngl, you can try using
this method, to make Moderngl work.

There are essentially two ways,

* Compiling Mesa yourselves see https://docs.mesa3d.org/install.html.
* Using msys2, which provides pre-compiled Mesa binaries.

Using MSYS2
___________

* Download and Install https://www.msys2.org/#installation
* Check whether you have 32-bit or 64-bit python.


32-bit python
+++++++++++++

If you have 32-bit python, then open ``C:\msys64\mingw32.exe`` and type the following

.. code-block:: sh

    pacman -S mingw-w64-i686-mesa



It will install mesa and its dependencies. Then you can add ``C:\msys64\mingw32\bin``
to PATH before ``C:\Windows`` and moderngl should be working. Also, you should set
an environment variable called ``GLCONTEXT_WIN_LIBGL`` which contains the path to opengl32
dll from mesa. In this case it should be ``GLCONTEXT_WIN_LIBGL=C:\msys64\mingw32\bin\opengl32.dll``.


64-bit python
+++++++++++++

If you have 64-bit python, then open ``C:\msys64\mingw64.exe`` and type the following

.. code-block:: sh

    pacman -S mingw-w64-x86_64-mesa

It will install mesa and it's dependencies. Then you can add ``C:\msys64\mingw64\bin`` to PATH before
``C:\Windows`` and moderngl should be working. Also, you should set an environment variable called
``GLCONTEXT_WIN_LIBGL`` which contains the path to opengl32
dll from mesa. In this case it should be ``GLCONTEXT_WIN_LIBGL=C:\msys64\mingw64\bin\opengl32.dll``


--------------------------------------------------------------------------------


.. File: install/using-moderngl-in-ci.rst


Using ModernGL in CI
====================

Windows CI Configuration
------------------------

ModernGL can't be run directly on Windows CI without the use of `Mesa`_. To get ModernGL running
you should first install Mesa from the `MSYS2 project`_ and adding it to the ``PATH``.

Steps
_____

1. Usually `MSYS2 project`_ should be installed by default by your CI provider in ``C:\msys64``. You 
   can refer `the documentation <https://www.msys2.org/docs/ci/>`_ on how to get it installed and make 
   sure to update it.

2. Then login through bash and enter ``pacman -S --noconfirm mingw-w64-x86_64-mesa``.
    
   .. code-block:: shell
      
       C:\msys64\usr\bin\bash -lc "pacman -S --noconfirm mingw-w64-x86_64-mesa"
   
   This will install Mesa binary, which moderngl would be using.
    
3. Then add ``C:\msys64\mingw64\bin`` to ``PATH``.
    
  .. code-block:: powershell
   
    $env:PATH = "C:\msys64\mingw64\bin;$env:PATH"

  .. warning::
    
        Make sure to delete ``C:\msys64\mingw64\bin\python.exe`` if it exists because the python provided
        by them would then be added to Global and some unexpected things may happen.
     
4. Then set an environment variable ``GLCONTEXT_WIN_LIBGL=C:\msys64\mingw64\bin\opengl32.dll``. This will
   make glcontext use ``C:\msys64\mingw64\bin\opengl32.dll`` for opengl drivers.

5. Then you can run moderngl as you want to.

.. _Mesa: https://mesa3d.org/
.. _MSYS2 project: https://www.msys2.org/

Example Configuration
_____________________

A example configuration for Github Actions:

.. code-block:: yaml

    name: Hello World
    on: [push, pull_request]

    jobs:
      build:
        runs-on: windows-latest
        steps:
          - uses: actions/checkout@v2
          - name: Set up Python
            uses: actions/setup-python@v2
            with:
              python-version: 3.9
          - uses: msys2/setup-msys2@v2
            with:
              msystem: MINGW64
              release: false
              install: mingw-w64-x86_64-mesa
          - name: Test using ModernGL
            shell: pwsh
            run: |
              Remove-Item C:\msys64\mingw64\bin\python.exe -Force
              $env:GLCONTEXT_WIN_LIBGL = "C:\msys64\mingw64\bin\opengl32.dll"
              python -m pip install -r requirements.txt
              python -m pytest
              
Linux
-----

For running ModernGL on Linux CI, you would need to configure ``xvfb`` so that it starts a Window in the background.
After that, you should be able to use ModernGL directly.

Steps
_____

1. Install ``xvfb`` from Package Manager.

   .. code-block:: bash
        
        sudo apt-get -y install xvfb

2. The run the below command, to start Xvfb from background.

   .. code-block:: bash
    
        sudo /usr/bin/Xvfb :0 -screen 0 1280x1024x24 &

3. You can run ModernGL now.

Example Configuration
_____________________

A example configuration for Github Actions:

.. code-block:: yaml

    name: Hello World
    on: [push, pull_request]

    jobs:
      build:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v2
          - name: Set up Python
            uses: actions/setup-python@v2
            with:
              python-version: 3.9
          - name: Prepare
            run: |
                sudo apt-get -y install xvfb
                sudo /usr/bin/Xvfb :0 -screen 0 1280x1024x24 &            
          - name: Test using ModernGL
            run: |
              python -m pip install -r requirements.txt
              python -m pytest

macOS
-----

You won't need any special configuration to run on macOS.



--------------------------------------------------------------------------------


.. File: the_guide/intro.rst

.. py:currentmodule:: moderngl

An introduction to OpenGL
=========================

The simplified story
--------------------

`OpenGL`_ (Open Graphics Library) has a long history reaching
all the way back to 1992 when it was created by `Silicon Graphics`_.
It was partly based in their proprietary `IRIS GL`_ 
(Integrated Raster Imaging System Graphics Library) library.

Today OpenGL is managed by the `Khronos Group`_, an open 
industry consortium of over 150 leading hardware and software
companies creating advanced, royalty-free, acceleration
standards for 3D graphics, Augmented and Virtual Reality,
vision and machine learning

The purpose of `OpenGL`_ is to provide a standard way to interact
with the graphics processing unit to achieve hardware accelerated rendering
across several platforms. How this is done under the hood is up to the
vendors (AMD, Nvidia, Intel, ARM .. etc) as long as the the specifications are
followed.

`OpenGL`_ has gone though many versions and it can be confusing when looking
up resources. Today we separate "Old OpenGL" and "Modern OpenGL".
From 2008 to 2010 version 3.x of OpenGL evolved until version
3.3 and 4.0 was released simultaneously.

In 2010 version 3.3, 4.0 and 4.1 was released to modernize the api
(simplified explanation) creating something that would be able
to utilize Direct3D 11-class hardware. **OpenGL 3.3 is the first
"Modern OpenGL" version** (simplified explanation). Everything
from this version is forward compatible all the way to the latest
4.x version. An optional deprecation mechanism was introduced to
disable outdated features. Running OpenGL in **core mode** would
remove all old features while running in **compatibility mode**
would still allow mixing the old and new api.

.. Note:: OpenGL 2.x, 3.0, 3.1 and 3.2 can of course access some
          modern OpenGL features directly, but for simplicity we are
          are focused on version 3.3 as it created the final
          standard we are using today. Older OpenGL was also
          a pretty wild world with countless vendor specific
          extensions. Modern OpenGL cleaned this up quite a bit.

In OpenGL we often talk about the **Fixed Pipeline** and the
**Programmable Pipeline**.

OpenGL code using the **Fixed Pipeline** (Old OpenGL) would use functions like
``glVertex``, ``glColor``, ``glMaterial`` ``glMatrixMode``,
``glLoadIdentity``, ``glBegin``, ``glEnd``, ``glVertexPointer``,
``glColorPointer``, ``glPushMatrix`` and ``glPopMatrix``.
The api had strong opinions and limitations on what you
could do, hiding what really went on under the hood.

OpenGL code using the **Programmable Pipeline** (Modern OpenGL) would use
functions like ``glCreateProgram``, ``UseProgram``. ``glCreateShader``,
``VertexAttrib*``, ``glBindBuffer*``, and ``glUniform*``.
This API mainly works with buffers of data and smaller programs
called "shaders" running on the GPU to process this data
using the **OpenGL Shading Language (GLSL)**. This gives
enormous flexibility but requires that we understand the
OpenGL pipeline (actually not that complicated).

Beyond OpenGL
-------------

OpenGL has a lot of "baggage" after 25 years and hardware has
drastically changed since its inception. Plans for "OpenGL 5"
was started as the **Next Generation OpenGL Initiative (glNext)**.
This Turned into the `Vulkan`_ API and was a grounds-up redesign
to unify OpenGL and OpenGL ES into one common API that will not be
backwards compatible with existing OpenGL versions.

This doesn't mean OpenGL is not worth learning today. In fact
learning 3.3+ shaders and understanding the rendering pipeline
will greatly help you understand `Vulkan`_. In most cases you can
pretty much copy paste the shaders over to `Vulkan`_.

Where does ModernGL fit into all this?
--------------------------------------

The ModernGL library exposes the **Programmable Pipeline**
using OpenGL 3.3 core or higher. However, we don't expose OpenGL
functions directly. Instead we expose features though various
objects like :py:class:`Buffer` and :py:class:`Program`
in a much more "pythonic" way. It's in other words a higher level
wrapper making OpenGL much easier to reason with. We try to hide
most of the complicated details to make the user more productive.
There are a lot of pitfalls with OpenGL and we remove most of them.

Learning ModernGL is more about learning shaders and the OpenGL
pipeline.

.. _Vulkan: https://www.khronos.org/vulkan/
.. _IRIS GL: https://wikipedia.org/wiki/IRIS_GL
.. _OpenGL: https://en.wikipedia.org/wiki/OpenGL
.. _Silicon Graphics: https://wikipedia.org/wiki/Silicon_Graphics
.. _Khronos Group: https://www.khronos.org


--------------------------------------------------------------------------------


.. File: the_guide/functionality_expansion/window_using.rst

Rendering to a window
=====================

.. py:currentmodule:: moderngl

By default, ModernGL does not have a window, but the `moderngl-window`_ module allows you to use one. The installation is as follows::

    pip install moderngl-window

`moderngl-window`_ uses `pyglet`_ as its default backend. It is installed automatically along with `moderngl-window`_. However, its use is limited to the supported functionality in ``moderngl_window.WindowConfig``.

.. rubric:: Entire source

.. literalinclude:: window_using_example.py
    :emphasize-lines: 1,4-12,57,62,64
    :linenos:

You can read the full usage of `moderngl-window`_ in its `documentation`_.

.. _moderngl-window: https://github.com/moderngl/moderngl-window
.. _documentation: https://moderngl-window.readthedocs.io/
.. _pyglet: https://github.com/pyglet/pyglet


--------------------------------------------------------------------------------


.. File: the_guide/getting_started/triangles_draw/uniform.rst

.. py:currentmodule:: moderngl

Uniform
=======

At some point you will need to make a lot of small changes to your rendering. Changing the screen aspect ratio, viewing angle, changing perspective/orthographic projection and much more. And in many situations it will be very convenient to use :py:class:`Uniform` -s.

:py:class:`Uniform` -s can be specified and used at any time during rendering. They allow you to replace all constants in the shader with variables and change them as needed. The :py:class:`Uniform` is initialized in the shader as follows::

    uniform float var1;

Changing the :py:class:`Uniform` -s value in ModernGL is very easy. For example, setting the value for our variable ``140.02`` is done as follows::

    vao.program['var1'].value = 140.02
    
    # or (using `__setitem__` shortcut)
    vao.program['var1'] = 140.02

If the variable type is not ``float``, but ``vec4``, simply list the values separated by commas::

    vao.program['var2'] = 1, 2, 3, 4
    # or
    vao.program['var2'] = [1, 2, 3, 4]
    # or
    vao.program['var2'] = (1, 2, 3, 4)

You need to list as many values as the :doc:`value type <../low_start/shader_basics>` takes: ``float`` will take 1 number, ``vec2`` will take 2 numbers, ``vec4`` will take 4 numbers, ``mat4`` will take 16 numbers, etc.

Let's consider a case where we need to change the size of our triangle. Take the :doc:`original triangle drawing code <../triangles_draw/one_familiar_triangle>` and make the following changes.

To change the scale (size) of the triangle, add a ``scale`` :py:class:`Uniform`. In the vertex shader it will be multiplied by all vertices and thus allow us to control the size of all triangles.

.. rubric:: Entire source
.. literalinclude:: triangle_draw_uniform.py
    :emphasize-lines: 15, 21, 54
    :linenos:

We set the scale value to ``2.0``, which means our triangle will be enlarged by 2 times.

.. figure:: enlarged_triangle.png
    :alt: Enlarged triangle
    :align: center
    :figclass: align-center

    Enlarged triangle

Now let's set the scale value to ``0.5`` to reduce the triangle by 2 times::

    vao.program['scale'] = 0.5

.. figure:: reduced_triangle.png
    :alt: Reduced triangle
    :align: center
    :figclass: align-center

    Reduced triangle

Uniforms can not only be set, but also read. This is done as follows::

    scale = vao.program['scale'].value

Also Uniforms can be written or read directly, in the form of bytes::

    # write
    scale = 2
    b_scale = numpy.asarray([scale], dtype='f4').tobytes()
    vao.program['scale'].write(b_scale)

    # read
    b_scale = vao.program['scale'].read()
    scale = numpy.frombuffer(b_scale, dtype='f4')[0]

    # `numpy.frombuffer()` converts a byte string into an array,
    # since we have one number, we select it from the array.

In most cases, directly using :py:class:`Uniform` -s ``.read()/.write()`` methods can speed up the code, but constantly manually converting variables into bytes does not make sense, since ModernGL already does it in the most optimized way.


--------------------------------------------------------------------------------


.. File: the_guide/getting_started/triangles_draw/one_familiar_triangle.rst

.. py:currentmodule:: moderngl

One familiar triangle
=====================

As with any graphics library guide, we also have a guide on how to draw a triangle.
Below is a slightly modified line drawing code from the :doc:`previous tutorial <../first_rendering/rendering>`. The following code draws one triangle:

.. rubric:: Entire source

.. literalinclude:: triangle_draw.py
    :emphasize-lines: 35-41, 43, 51
    :linenos:

When you run the code you will see this:

.. figure:: triangle.png
    :alt: Triangle
    :align: center
    :figclass: align-center

    Triangle

As you may have noticed, we only specified three colors for each vertex, but OpenGL interpolates our triangle and we see a soft transition of colors.

At this point you can try out the fragment shader, for example, let's draw a lighting effect:

.. rubric:: Fragment shader

.. literalinclude:: lighted.frag.glsl
    :language: glsl
    :emphasize-lines: 8-9

.. figure:: lighted_triangle.png
    :alt: Lighted triangle
    :align: center
    :figclass: align-center

    Lighted triangle
    
Shaders are not very fond of branching algorithms and operations such as ``if (condition) {action} else {action}``. It is recommended to use formulas more often.



--------------------------------------------------------------------------------


.. File: the_guide/getting_started/low_start/creating_context.rst

.. py:currentmodule:: moderngl

Creating a Context
==================

Before we can do anything with ModernGL we need a :py:class:`Context`.
The :py:class:`Context` object makes us able to create OpenGL resources.
ModernGL can only create headless contexts (no window), but it can also detect
and use contexts from a large range of window libraries. The `moderngl-window`_
library is a good start or reference for rendering to a window.

Most of the example code here assumes a ``ctx`` variable exists with a
headless context::

    # standalone=True makes a headless context
    ctx = moderngl.create_context(standalone=True)

Detecting an active context created by a window library is simply::

    ctx = moderngl.create_context()

More details about context creation can be found in the :ref:`context`
section.

.. _moderngl-window: https://github.com/moderngl/moderngl-window


--------------------------------------------------------------------------------


.. File: the_guide/getting_started/low_start/moderngl_types.rst

.. py:currentmodule:: moderngl

ModernGL Types
==============

Before throwing you into doing shaders we'll go through some of the
most important types/objects in ModernGL.

* :py:class:`Buffer` is an OpenGL buffer we can for example write
  vertex data into. This data will reside in graphics memory.
* :py:class:`Program` is a shader program. We can feed it GLSL
  source code as strings to set up our shader program
* :py:class:`VertexArray` is a light object responsible for
  communication between :py:class:`Buffer` and :py:class:`Program`
  so it can understand how to access the provided buffers
  and do the rendering call.
  These objects are currently immutable but are cheap to make.
* :py:class:`Texture`, :py:class:`TextureArray`, :py:class:`Texture3D`
  and :py:class:`TextureCube` represents the different texture types.
  :py:class:`Texture` is a 2d texture and is most commonly used.
* :py:class:`Framebuffer` is an offscreen render target. It supports
  different attachments types such as a :py:class:`Texture`
  and a depth texture/buffer.

All of the objects above can only be created from a :py:class:`Context` object:

* :py:meth:`Context.buffer`
* :py:meth:`Context.program`
* :py:meth:`Context.vertex_array`
* :py:meth:`Context.texture`
* :py:meth:`Context.texture_array`
* :py:meth:`Context.texture3d`
* :py:meth:`Context.texture_cube`
* :py:meth:`Context.framebuffer`

The ModernGL types cannot be extended as in; you cannot subclass them.
Extending them must be done through substitution and not inheritance.
This is related to performance. Most objects have an ``extra``
property that can contain any python object.


--------------------------------------------------------------------------------


.. File: the_guide/getting_started/low_start/shader_transform.rst

.. py:currentmodule:: moderngl

Vertex Shader (transforms)
==========================

Let's get our hands dirty right away and jump into it by showing the
simplest forms of shaders in OpenGL. These are called transforms or
transform feedback. Instead of drawing to the screen we simply
capture the output of a shader into a :py:class:`Buffer`.

The example below shows shader program with only a vertex shader.
It has no input data, but we can still force it to run N times.
The ``gl_VertexID`` (int) variable is a built-in value in vertex
shaders containing an integer representing the vertex number
being processed.

Input variables in vertex shaders are called **attributes**
(we have no inputs in this example)
while output values are called **varyings**.

.. code::

    import struct
    import moderngl

    ctx = moderngl.create_context(standalone=True)

    program = ctx.program(
        vertex_shader="""
        #version 330

        // Output values for the shader. They end up in the buffer.
        out float value;
        out float product;

        void main() {
            // Implicit type conversion from int to float will happen here
            value = gl_VertexID;
            product = gl_VertexID * gl_VertexID;
        }
        """,
        # What out varyings to capture in our buffer!
        varyings=["value", "product"],
    )

    NUM_VERTICES = 10

    # We always need a vertex array in order to execute a shader program.
    # Our shader doesn't have any buffer inputs, so we give it an empty array.
    vao = ctx.vertex_array(program, [])

    # Create a buffer allocating room for 20 32 bit floats
    # num of vertices (10) * num of varyings per vertex (2) * size of float in bytes (4)
    buffer = ctx.buffer(reserve=NUM_VERTICES * 2 * 4)

    # Start a transform with buffer as the destination.
    # We force the vertex shader to run 10 times
    vao.transform(buffer, vertices=NUM_VERTICES)

    # Unpack the 20 float values from the buffer (copy from graphics memory to system memory).
    # Reading from the buffer will cause a sync (the python program stalls until the shader is done)
    data = struct.unpack("20f", buffer.read())
    for i in range(0, 20, 2):
        print("value = {}, product = {}".format(*data[i:i+2]))

Output of the program is::

    value = 0.0, product = 0.0
    value = 1.0, product = 1.0
    value = 2.0, product = 4.0
    value = 3.0, product = 9.0
    value = 4.0, product = 16.0
    value = 5.0, product = 25.0
    value = 6.0, product = 36.0
    value = 7.0, product = 49.0
    value = 8.0, product = 64.0
    value = 9.0, product = 81.0

The GPU is at the very least slightly offended by the meager amount
work we assigned it, but this at least shows the basic concept of transforms.
We would in most situations also not read the results back into
system memory because it's slow, but sometimes it is needed.

This shader program could for example be modified to generate some
geometry or data for any other purpose you might imagine useful.
Using modulus (``mod(number, divisor)``) on ``gl_VertexID`` can get you pretty far.

.. Warning::
    
    One known bug in many OpenGL drivers is related to the ``mod(number, divisor)`` function. It lies in the fact that if the first argument is exactly 2 times the second, then instead of ``0`` you will receive the value of the second argument [divisor]. To avoid this error, it is recommended to use the following additional function (insert it into the shader before ``void main()`` and then always call ``mod2()`` instead of ``mod()``)::
    
        float mod2(float number, float divisor) {
            float result = mod(number, divisor);
            if (result >= divisor) {
                result = 0.0;
            }
            return result;
        }



--------------------------------------------------------------------------------


.. File: the_guide/getting_started/low_start/shader_basics.rst

.. py:currentmodule:: moderngl

Shader Introduction
===================

Shaders are small programs running on the `GPU`_ (Graphics Processing Unit).
We are using a fairly simple language called `GLSL`_ (OpenGL Shading Language).
This is a C-style language, so it covers most of the features you would expect
with such a language. Control structures (for-loops, if-else statements, etc)
exist in GLSL, including the switch statement.

.. Note:: The name "shader" comes from the fact that these small GPU programs was
          originally created for shading (lighting) 3D scenes. This started
          as per-vertex lighting when the early shaders could only process 
          vertices and evolved into per-pixel lighting when the fragment
          shader was introduced.
          They are used in many other areas today, but the name have stuck around.

Examples of types are::

    bool value = true;
    int value = 1;
    uint value = 1;
    float value = 0.0;
    double value = 0.0;

Each type above also has a 2, 3 and 4 component version::

    // float (default) type
    vec2 value = vec2(0.0, 1.0);
    vec3 value = vec3(0.0, 1.0, 2.0);
    vec4 value = vec4(0.0);

    // signed and unsigned integer vectors
    ivec3 value = ivec3(0);
    uvec3 value = ivec3(0);
    // etc ..

More about GLSL `data types`_ can be found in the Khronos wiki.

The available functions are for example: ``radians``, ``degrees``
``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``pow``
``exp``, ``log``, ``exp2``, ``log2``, ``sqrt``, ``inversesqrt``,
``abs``, ``sign``, ``floor``, ``ceil``, ``fract``, ``mod``,
``min``, ``max``, ``clamp``, ``mix``, ``step``, ``smoothstep``,
``length``, ``distance``, ``dot``, ``cross``, ``normalize``,
``faceforward``, ``reflect``, ``refract``, ``any``, ``all`` etc.

All functions can be found in the `OpenGL Reference Page`_ 
(exclude functions starting with ``gl``).
Most of the functions exist in several overloaded versions
supporting different data types.

The basic setup for a shader is the following::

    #version 330

    void main() {
    }

The ``#version`` statement is mandatory and should at least be 330
(GLSL version 3.3 matching OpenGL version 3.3). The version statement
**should always be the first line in the source code**.
Higher version number is only needed if more fancy features are needed.
By the time you need those you probably know what you are doing.

What we also need to realize when working with shaders is that
they are executed in parallel across all the cores on your GPU.
This can be everything from tens, hundreds, thousands or more
cores. Even integrated GPUs today are very competent.

For those
who have not worked with shaders before it can be mind-boggling
to see the work they can get done in a matter of microseconds.
All shader executions / rendering calls are also asynchronous
running in the background while your python code is doing
other things (but certain operations can cause a "sync" stalling
until the shader program is done).

Let's try to use the shader in the :doc:`simplest way (next step) <shader_transform>`.

.. _GPU: https://wikipedia.org/wiki/Graphics_processing_unit
.. _GLSL: https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language
.. _data types: https://www.khronos.org/opengl/wiki/Data_Type_(GLSL)
.. _OpenGL Reference Page: https://www.khronos.org/registry/OpenGL-Refpages/gl4/


--------------------------------------------------------------------------------


.. File: the_guide/getting_started/first_rendering/program.rst

.. py:currentmodule:: moderngl

Program
=======

ModernGL is different from standard plotting libraries.
You can define your own shader program to render stuff.
This could complicate things, but also provides freedom on how you render
your data.

Previously, we looked at creating a :doc:`vertex shader-only program <../low_start/shader_transform>` that could only read data from the input buffer and write the converted data to the output buffer. Now let's add a fragment shader to our program; it will allow us to create an algorithm for writing pixels into a texture, that is, perform the :doc:`main function <../low_start/shader_basics>` of the shader.

Here is a sample program that passes the input vertex coordinates as is to
screen coordinates.

Screen coordinates are in the [-1, 1], [-1, 1] range for x and y axes.
The (-1, -1) point is the lower left corner of the screen.

.. figure:: screen_coordinates.png
    :alt: Screen Coordinates
    :align: center
    :figclass: align-center

    The screen coordinates

The program will also process a color information.

.. rubric:: Entire source

.. literalinclude:: first.1.py
    :emphasize-lines: 5-
    :linenos:

.. rubric:: Vertex Shader

.. literalinclude:: first.1.py
    :language: glsl
    :dedent: 8
    :lines: 7-17

.. rubric:: Fragment Shader

.. literalinclude:: first.1.py
    :language: glsl
    :dedent: 8
    :lines: 20-28

Proceed to the :doc:`next step <buffer>`.


--------------------------------------------------------------------------------


.. File: the_guide/getting_started/first_rendering/vertex_array.rst

.. py:currentmodule:: moderngl

Vertex Array
============

:py:class:`VertexArray` is something like a pipeline, where as arguments :py:meth:`Context.vertex_array` takes a :py:class:`Program`, a :py:class:`Buffer` with input data, and the names of input variables for this program.

:py:class:`VertexArray` in ModernGL can be initialized in two ways: one buffer for all input variables or multiple buffers for all input variables.

One input buffer for all input variables (simple :py:class:`VertexArray` version)::
    
    vao = ctx.vertex_array(program, buffer, 'input_var1', 'input_var2')

Multiple input buffers for all input variables::
    
    vao = ctx.vertex_array(
        program,
        [
            (buffer1, '3f 2f', 'input_var1', 'input_var2'),
            (buffer2, '4f', 'input_var3')
        ]
    )
    
You can understand ``'3f 2f'`` and ``'4f'`` as the type of the input variable (or variables), that is, 3 floats and 2 floats, which form, for example, ``vec3 + vec2`` and 4 floats, which form ``vec4``.


In our example we use a simple implementation of this method.

.. rubric:: Entire source

.. literalinclude:: first.3.py
    :emphasize-lines: 42
    :linenos:

Proceed to the :doc:`next step <rendering>`.


--------------------------------------------------------------------------------


.. File: the_guide/getting_started/first_rendering/buffer.rst

.. py:currentmodule:: moderngl

Buffer
======

:py:class:`Buffer` is a dedicated area of GPU memory. Can store anything in bytes.

Creating a buffer is essentially allocating an area of memory into which data is later written. Therefore, when creating an empty buffer, it may contain memory fragments of deleted objects. Example of creating an empty buffer::

    buf = ctx.buffer(reserve=1)

The ``reserve`` parameter specifies how many bytes should be allocated, the minimum number is 1.

The buffer is cleared by writing zeros to the allocated memory area. However, this must be done manually::

    buf.clear()

ModernGL allows you to create 2 types of buffer: dynamic and non-dynamic. To do this, when creating a buffer, use the keyword argument ``dynamic=True/False`` (by default ``False``)::

    buf = ctx.buffer(reserve=32, dynamic=True)

.. Note::

    Using the ``dynamic=True`` parameter tells the GPU that actions with this :py:class:`Buffer` will be performed very often. This parameter is optional, but is recommended if the :py:class:`Buffer` is used frequently.

Later, using the :py:meth:`Buffer.orphan` function, you can change the buffer size at any time::

    buf.orphan(size=64)

After changing the buffer size, you will need to write data there. This is done using the :py:meth:`Buffer.write` function. This function will write data from RAM to GPU memory::

    buf.write(b'any bytes data')

However, if the size of this buffer was changed after it was added to :py:class:`VertexArray`, then when calling :py:meth:`VertexArray.render()` you will need to specify the new number of vertices in the ``vertices`` parameter. For example::

    # If from the contents of this buffer every 12 bytes fall on one vertex.
    vao.render(vertices=buf.size // 4 // 3)

The same will need to be done when calling the :py:meth:`VertexArray.transform` function::

    # If from the contents of this buffer every 12 bytes fall on one vertex.
    # output_buf - the buffer into which the transformation will be performed.
    vao.transform(output_buf, vertices=buf.size // 4 // 3)

After :py:meth:`VertexArray.transform` writes data to ``output_buf`` using a :doc:`vertex shader <../low_start/shader_transform>`, you may need to read it — use :py:meth:`Buffer.read`. This function will read data from GPU memory into RAM::

    bytes_data = buf.read()

.. Warning::

    Transferring data between RAM and GPU memory comes at a huge performance cost. It is recommended to use :py:meth:`Buffer.write` and :py:meth:`Buffer.read` as little as possible.

    If you just need to copy data between buffers, look towards the :py:meth:`Context.copy_buffer` function::

        ctx.copy_buffer(destination_buf, source_buf)

In our example, we simply create a static buffer and write data immediately when it is created::

    vbo = ctx.buffer(vertices.astype("f4").tobytes())

We called it `VBO`_ (Vertex Buffer Object) because we will store vertex data in this buffer.

.. Sidebar::

    .. Note::
        For the convenience of transferring data to the GPU memory [in a dedicated :py:class:`Buffer` area], here we use the `NumPy`_ library.

        `NumPy`_ installation::

            pip install numpy

        If you want fewer dependencies, you can try Python's built-in `struct`_ module with the ``struct.pack()`` and ``struct.unpack()`` methods.

.. rubric:: Entire source

.. literalinclude:: first.2.py
    :emphasize-lines: 2,33-
    :linenos:

Proceed to the :doc:`next step <vertex_array>`.

.. _NumPy: https://github.com/numpy/numpy
.. _struct: https://docs.python.org/3/library/struct.html
.. _VBO: https://www.khronos.org/opengl/wiki/Vertex_Specification#Vertex_Buffer_Object


--------------------------------------------------------------------------------


.. File: the_guide/getting_started/first_rendering/rendering.rst

.. py:currentmodule:: moderngl

Standalone rendering
====================

Standalone (offline) rendering allows you to render without using a window, and is included in ModernGL by default.

Rendering occurs when :py:meth:`VertexArray.render` is called. By default, the ``mode`` parameter is :py:attr:`moderngl.TRIANGLES`, but since we need to draw a line, we change the mode value to :py:attr:`moderngl.LINES`::

    vao.render(moderngl.LINES)  # "mode" is the first optional argument

To display the rendering result, we use the `Pillow (PIL)`_ library that comes with Python. Let's return the texture from the GPU memory to RAM and call the ``PIL.Image.show()`` method to show it.

.. rubric:: Entire source

.. literalinclude:: first.4.py
    :emphasize-lines: 4,46-
    :linenos:

The result will be something like this:

.. figure:: rendering_result.png
    :alt: Rendering result
    :align: center
    :figclass: align-center

    Rendering result

.. _Pillow (PIL): https://pillow.readthedocs.io/


--------------------------------------------------------------------------------


.. File: techniques/headless_ubuntu_18_server.rst


Headless on Ubuntu 18 Server
============================

Dependencies
------------

Headless rendering can be achieved with EGL or X11.
We'll cover both cases.

Starting with fresh ubuntu 18 server install we need to install required
packages::

    sudo apt-install python3-pip mesa-utils libegl1-mesa xvfb

This should install mesa an diagnostic tools if needed later.

* ``mesa-utils`` installs libgl1-mesa and tools like ``glxinfo```
* ``libegl1-mesa`` is optional if using EGL instead of X11

Creating a context
------------------

The libraries we are going to interact with has the following locations::

    /usr/lib/x86_64-linux-gnu/libGL.so.1
    /usr/lib/x86_64-linux-gnu/libX11.so.6
    /usr/lib/x86_64-linux-gnu/libEGL.so.1

Double check that you have these libraries installed. ModernGL
through the glcontext library will use ``ctype.find_library``
to locate the latest installed version.

Before we can create a context we to run a virtual display::

    export DISPLAY=:99.0
    Xvfb :99 -screen 0 640x480x24 &

Now we can create a context with x11 or egl:

.. code::

    # X11
    import moderngl
    ctx = moderngl.create_context(
        standalone=True,
        # These are OPTIONAL if you want to load a specific version
        libgl='libGL.so.1',
        libx11='libX11.so.6',
    )

    # EGL
    import moderngl
    ctx = moderngl.create_context(
        standalone=True,
        backend='egl',
        # These are OPTIONAL if you want to load a specific version
        libgl='libGL.so.1',
        libegl='libEGL.so.1',
    )


Running an example
------------------

Checking that everything works can be done with a basic triangle example.

Install dependencies::

    pip3 install moderngl numpy pyrr pillow

The following example renders a triangle and writes
it to a png file so we can verify the contents.

.. image:: output.png

.. code:: python

    import moderngl
    import numpy as np
    from PIL import Image
    from pyrr import Matrix44

    # -------------------
    # CREATE CONTEXT HERE
    # -------------------

    prog = ctx.program(vertex_shader="""
        #version 330
        uniform mat4 model;
        in vec2 in_vert;
        in vec3 in_color;
        out vec3 color;
        void main() {
            gl_Position = model * vec4(in_vert, 0.0, 1.0);
            color = in_color;
        }
        """,
        fragment_shader="""
        #version 330
        in vec3 color;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(color, 1.0);
        }
    """)

    vertices = np.array([
        -0.6, -0.6,
        1.0, 0.0, 0.0,
        0.6, -0.6,
        0.0, 1.0, 0.0,
        0.0, 0.6,
        0.0, 0.0, 1.0,
    ], dtype='f4')

    vbo = ctx.buffer(vertices)
    vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_color')
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((512, 512), 4)])

    fbo.use()
    ctx.clear()
    prog['model'].write(Matrix44.from_eulers((0.0, 0.1, 0.0), dtype='f4'))
    vao.render(moderngl.TRIANGLES)

    data = fbo.read(components=3)
    image = Image.frombytes('RGB', fbo.size, data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save('output.png')



--------------------------------------------------------------------------------


.. File: reference/sampler.rst

Sampler
=======

.. py:class:: Sampler

    Returned by :py:meth:`Context.sampler`

    A Sampler Object is an OpenGL Object that stores the sampling parameters for a Texture access inside of a shader.

    When a sampler object is bound to a texture image unit,
    the internal sampling parameters for a texture bound to the same image unit are all ignored.
    Instead, the sampling parameters are taken from this sampler object.

    Unlike textures, a samplers state can also be changed freely be at any time
    without the sampler object being bound/in use.

    Samplers are bound to a texture unit and not a texture itself. Be careful with leaving
    samplers bound to texture units as it can cause texture incompleteness issues
    (the texture bind is ignored).

    Sampler bindings do clear automatically between every frame so a texture unit
    need at least one bind/use per frame.

Methods
-------

.. py:method:: Sampler.use(location: int = 0) -> None:

    Bind the sampler to a texture unit.

    :param int location: The texture unit

.. py:method:: Sampler.clear(location: int = 0) -> None:

    Clear the sampler binding on a texture unit.

    :param int location: The texture unit

.. py:method:: Sampler.assign(index: int) -> tuple

    Helper method for assigning samplers to scopes.

    Example::

        s1 = ctx.sampler(...)
        s2 = ctx.sampler(...)
        ctx.scope(samplers=(s1.assign(0), s1.assign(1)), ...)mpler

.. py:method:: Sampler.release

Attributes
----------

.. py:attribute:: Sampler.texture
    :type: Texture

    The texture object to sample.

.. py:attribute:: Sampler.repeat_x
    :type: bool

    The x repeat flag for the sampler (Default ``True``).

    Example::

        # Enable texture repeat (GL_REPEAT)
        sampler.repeat_x = True

        # Disable texture repeat (GL_CLAMP_TO_EDGE)
        sampler.repeat_x = False

.. py:attribute:: Sampler.repeat_y
    :type: bool

    The y repeat flag for the sampler (Default ``True``).

    Example::

        # Enable texture repeat (GL_REPEAT)
        sampler.repeat_y = True

        # Disable texture repeat (GL_CLAMP_TO_EDGE)
        sampler.repeat_y = False

.. py:attribute:: Sampler.repeat_z
    :type: bool

    The z repeat flag for the sampler (Default ``True``).

    Example::

        # Enable texture repeat (GL_REPEAT)
        sampler.repeat_z = True

        # Disable texture repeat (GL_CLAMP_TO_EDGE)
        sampler.repeat_z = False

.. py:attribute:: Sampler.filter
    :type: tuple

    The minification and magnification filter for the sampler.

    (Default ``(moderngl.LINEAR. moderngl.LINEAR)``)

    Example::

        sampler.filter == (moderngl.NEAREST, moderngl.NEAREST)
        sampler.filter == (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        sampler.filter == (moderngl.NEAREST_MIPMAP_LINEAR, moderngl.NEAREST)
        sampler.filter == (moderngl.LINEAR_MIPMAP_NEAREST, moderngl.NEAREST)

.. py:attribute:: Sampler.compare_func
    :type: tuple

    The compare function for a depth textures (Default ``'?'``).

    By default samplers don't have depth comparison mode enabled.
    This means that depth texture values can be read as a ``sampler2D``
    using ``texture()`` in a GLSL shader by default.

    When setting this property to a valid compare mode, ``GL_TEXTURE_COMPARE_MODE``
    is set to ``GL_COMPARE_REF_TO_TEXTURE`` so that texture lookup
    functions in GLSL will return a depth comparison result instead
    of the actual depth value.

    Accepted compare functions::

        .compare_func = ''    # Disale depth comparison completely
        sampler.compare_func = '<='  # GL_LEQUAL
        sampler.compare_func = '<'   # GL_LESS
        sampler.compare_func = '>='  # GL_GEQUAL
        sampler.compare_func = '>'   # GL_GREATER
        sampler.compare_func = '=='  # GL_EQUAL
        sampler.compare_func = '!='  # GL_NOTEQUAL
        sampler.compare_func = '0'   # GL_NEVER
        sampler.compare_func = '1'   # GL_ALWAYS

.. py:attribute:: Sampler.anisotropy
    :type: float

    Number of samples for anisotropic filtering (Default ``1.0``).

    The value will be clamped in range ``1.0`` and ``ctx.max_anisotropy``.

    Any value greater than 1.0 counts as a use of anisotropic filtering::

        # Disable anisotropic filtering
        sampler.anisotropy = 1.0

        # Enable anisotropic filtering suggesting 16 samples as a maximum
        sampler.anisotropy = 16.0

.. py:attribute:: Sampler.border_color
    :type: tuple

    The (r, g, b, a) color for the texture border (Default ``(0.0, 0.0, 0.0, 0.0)``).

    When setting this value the ``repeat_`` values are overridden setting the texture wrap to return
    the border color when outside [0, 1] range.

    Example::

        # Red border color
        sampler.border_color = (1.0, 0.0, 0.0, 0.0)

.. py:attribute:: Sampler.min_lod
    :type: float

    Minimum level-of-detail parameter (Default ``-1000.0``).

    This floating-point value limits the selection of highest resolution mipmap (lowest mipmap level)

.. py:attribute:: Sampler.max_lod
    :type: float

    Minimum level-of-detail parameter (Default ``1000.0``).

    This floating-point value limits the selection of the lowest resolution mipmap (highest mipmap level)

.. py:attribute:: Sampler.ctx
    :type: Context

    The context this object belongs to

.. py:attribute:: Sampler.glo
    :type: int

    The internal OpenGL object.
    This values is provided for interoperability and debug purposes only.

.. py:attribute:: Sampler.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: reference/texture3d.rst

Texture3D
=========

.. py:class:: Texture3D

    Returned by :py:meth:`Context.texture3d`

    A Texture is an OpenGL object that contains one or more images that all have the same image format.

    A texture can be used in two ways. It can
    be the source of a texture access from a Shader, or it can be used
    as a render target.

    A Texture3D object cannot be instantiated directly, it requires a context.
    Use :py:meth:`Context.texture3d` to create one.

Methods
-------

.. py:method:: Texture3D.read
.. py:method:: Texture3D.read_into
.. py:method:: Texture3D.write
.. py:method:: Texture3D.build_mipmaps
.. py:method:: Texture3D.bind_to_image
.. py:method:: Texture3D.use
.. py:method:: Texture3D.release
.. py:method:: Texture3D.get_handle

Attributes
----------

.. py:attribute:: Texture3D.repeat_x
.. py:attribute:: Texture3D.repeat_y
.. py:attribute:: Texture3D.repeat_z
.. py:attribute:: Texture3D.filter
.. py:attribute:: Texture3D.swizzle
.. py:attribute:: Texture3D.width
.. py:attribute:: Texture3D.height
.. py:attribute:: Texture3D.depth
.. py:attribute:: Texture3D.size
.. py:attribute:: Texture3D.dtype
.. py:attribute:: Texture3D.components

.. py:attribute:: Texture3D.ctx
    :type: Context

    The context this object belongs to

.. py:attribute:: Texture3D.glo
    :type: int

    The internal OpenGL object.
    This values is provided for interoperability and debug purposes only.

.. py:attribute:: Texture3D.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: reference/renderbuffer.rst

Renderbuffer
============

.. py:class:: Renderbuffer

    Returned by :py:meth:`Context.renderbuffer` or :py:meth:`Context.depth_renderbuffer`

    Renderbuffer objects are OpenGL objects that contain images.

    They are created and used specifically with :py:class:`Framebuffer` objects.
    They are optimized for use as render targets, while :py:class:`Texture` objects
    may not be, and are the logical choice when you do not need to sample
    from the produced image. If you need to resample, use Textures instead.
    Renderbuffer objects also natively accommodate multisampling.

    A Renderbuffer object cannot be instantiated directly, it requires a context.
    Use :py:meth:`Context.renderbuffer` or :py:meth:`Context.depth_renderbuffer`
    to create one.

Methods
-------

.. py:method:: Renderbuffer.release

Attributes
----------

.. py:attribute:: Renderbuffer.width
    :type: int

    The width of the renderbuffer.

.. py:attribute:: Renderbuffer.height
    :type: int

    The height of the renderbuffer.

.. py:attribute:: Renderbuffer.size
    :type: Tuple[int, int]

    The size of the renderbuffer.

.. py:attribute:: Renderbuffer.samples
    :type: int

    The number of samples for multisampling.

.. py:attribute:: Renderbuffer.components
    :type: int

    The number components.

.. py:attribute:: Renderbuffer.depth
    :type: bool

    Determines if the renderbuffer contains depth values.

.. py:attribute:: Renderbuffer.dtype
    :type: str

    Data type.

.. py:attribute:: Renderbuffer.ctx
    :type: Context

    The context this object belongs to

.. py:attribute:: Renderbuffer.glo
    :type: int

    The internal OpenGL object.
    This values is provided for interoperability and debug purposes only.

.. py:attribute:: Renderbuffer.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: reference/moderngl.rst

moderngl
========

.. py:module:: moderngl

.. code-block:: python

    import moderngl

    window = ...
    ctx = moderngl.create_context()
    # store a ref to ctx

The module object itself is responsible for creating a :py:class:`Context` object.

.. py:function:: moderngl.create_context(require: int = 330, standalone: bool = False) -> Context

    Create a ModernGL context by loading OpenGL functions from an existing OpenGL context.
    An OpenGL context must exist. Call this after a window is created or opt for the windowless standalone mode.
    Other backend specific settings are passed as keyword arguments.

    Context sharing is known to not work properly, please avoid using it.
    There is a paramter `share` for that to attempt to create a shared context.

    :param int require: OpenGL version code
    :param bool standalone: Headless flag

    Example::

        # Accept the current context version
        ctx = moderngl.create_context()

        # Require at least OpenGL 4.3
        ctx = moderngl.create_context(require=430)

        # Create a windowless context
        ctx = moderngl.create_context(standalone=True)

.. py:function:: moderngl.create_standalone_context(...) -> Context

    Deprecated, use :py:func:`moderngl.create_context()` with the standalone parameter set.

.. py:function:: moderngl.get_context() -> Context

    Returns the previously created context object.

    Example::

        # my_app.py

        from moderngl import create_context

        ctx = create_context(...)

        # my_renderer.py

        from moderngl import get_context

        class MyRenderer:
            def __init__(self):
                self.ctx = get_context()
                self.program = ...
                self.vao = ...

Context Flags
-------------

These were moved to :py:class:`Context`.

.. py:attribute:: moderngl.NOTHING

    See :py:attr:`Context.NOTHING`

.. py:attribute:: moderngl.BLEND

    See :py:attr:`Context.BLEND`

.. py:attribute:: moderngl.DEPTH_TEST

    See :py:attr:`Context.DEPTH_TEST`

.. py:attribute:: moderngl.CULL_FACE

    See :py:attr:`Context.CULL_FACE`

.. py:attribute:: moderngl.RASTERIZER_DISCARD

    See :py:attr:`Context.RASTERIZER_DISCARD`

.. py:attribute:: moderngl.PROGRAM_POINT_SIZE

    See :py:attr:`Context.PROGRAM_POINT_SIZE`

.. py:attribute:: moderngl.POINTS

    See :py:attr:`Context.POINTS`

.. py:attribute:: moderngl.LINES

    See :py:attr:`Context.LINES`

.. py:attribute:: moderngl.LINE_LOOP

    See :py:attr:`Context.LINE_LOOP`

.. py:attribute:: moderngl.LINE_STRIP

    See :py:attr:`Context.LINE_STRIP`

.. py:attribute:: moderngl.TRIANGLES

    See :py:attr:`Context.TRIANGLES`

.. py:attribute:: moderngl.TRIANGLE_STRIP

    See :py:attr:`Context.TRIANGLE_STRIP`

.. py:attribute:: moderngl.TRIANGLE_FAN

    See :py:attr:`Context.TRIANGLE_FAN`

.. py:attribute:: moderngl.LINES_ADJACENCY

    See :py:attr:`Context.LINES_ADJACENCY`

.. py:attribute:: moderngl.LINE_STRIP_ADJACENCY

    See :py:attr:`Context.LINE_STRIP_ADJACENCY`

.. py:attribute:: moderngl.TRIANGLES_ADJACENCY

    See :py:attr:`Context.TRIANGLES_ADJACENCY`

.. py:attribute:: moderngl.TRIANGLE_STRIP_ADJACENCY

    See :py:attr:`Context.TRIANGLE_STRIP_ADJACENCY`

.. py:attribute:: moderngl.PATCHES

    See :py:attr:`Context.PATCHES`

.. py:attribute:: moderngl.NEAREST

    See :py:attr:`Context.NEAREST`

.. py:attribute:: moderngl.LINEAR

    See :py:attr:`Context.LINEAR`

.. py:attribute:: moderngl.NEAREST_MIPMAP_NEAREST

    See :py:attr:`Context.NEAREST_MIPMAP_NEAREST`

.. py:attribute:: moderngl.LINEAR_MIPMAP_NEAREST

    See :py:attr:`Context.LINEAR_MIPMAP_NEAREST`

.. py:attribute:: moderngl.NEAREST_MIPMAP_LINEAR

    See :py:attr:`Context.NEAREST_MIPMAP_LINEAR`

.. py:attribute:: moderngl.LINEAR_MIPMAP_LINEAR

    See :py:attr:`Context.LINEAR_MIPMAP_LINEAR`

.. py:attribute:: moderngl.ZERO

    See :py:attr:`Context.ZERO`

.. py:attribute:: moderngl.ONE

    See :py:attr:`Context.ONE`

.. py:attribute:: moderngl.SRC_COLOR

    See :py:attr:`Context.SRC_COLOR`

.. py:attribute:: moderngl.ONE_MINUS_SRC_COLOR

    See :py:attr:`Context.ONE_MINUS_SRC_COLOR`

.. py:attribute:: moderngl.SRC_ALPHA

    See :py:attr:`Context.SRC_ALPHA`

.. py:attribute:: moderngl.ONE_MINUS_SRC_ALPHA

    See :py:attr:`Context.ONE_MINUS_SRC_ALPHA`

.. py:attribute:: moderngl.DST_ALPHA

    See :py:attr:`Context.DST_ALPHA`

.. py:attribute:: moderngl.ONE_MINUS_DST_ALPHA

    See :py:attr:`Context.ONE_MINUS_DST_ALPHA`

.. py:attribute:: moderngl.DST_COLOR

    See :py:attr:`Context.DST_COLOR`

.. py:attribute:: moderngl.ONE_MINUS_DST_COLOR

    See :py:attr:`Context.ONE_MINUS_DST_COLOR`

.. py:attribute:: moderngl.DEFAULT_BLENDING

    See :py:attr:`Context.DEFAULT_BLENDING`

.. py:attribute:: moderngl.ADDITIVE_BLENDING

    See :py:attr:`Context.ADDITIVE_BLENDING`

.. py:attribute:: moderngl.PREMULTIPLIED_ALPHA

    See :py:attr:`Context.PREMULTIPLIED_ALPHA`

.. py:attribute:: moderngl.FUNC_ADD

    See :py:attr:`Context.FUNC_ADD`

.. py:attribute:: moderngl.FUNC_SUBTRACT

    See :py:attr:`Context.FUNC_SUBTRACT`

.. py:attribute:: moderngl.FUNC_REVERSE_SUBTRACT

    See :py:attr:`Context.FUNC_REVERSE_SUBTRACT`

.. py:attribute:: moderngl.MIN

    See :py:attr:`Context.MIN`

.. py:attribute:: moderngl.MAX

    See :py:attr:`Context.MAX`

.. py:attribute:: moderngl.FIRST_VERTEX_CONVENTION

    See :py:attr:`Context.FIRST_VERTEX_CONVENTION`

.. py:attribute:: moderngl.LAST_VERTEX_CONVENTION

    See :py:attr:`Context.LAST_VERTEX_CONVENTION`


--------------------------------------------------------------------------------


.. File: reference/texture.rst

Texture
=======

.. py:class:: Texture

    Returned by :py:meth:`Context.texture` and :py:meth:`Context.depth_texture`

    A Texture is an OpenGL object that contains one or more images that all have the same image format.

    A texture can be used in two ways. It can
    be the source of a texture access from a Shader, or it can be used
    as a render target.

    A Texture object cannot be instantiated directly, it requires a context.
    Use :py:meth:`Context.texture` or :py:meth:`Context.depth_texture`
    to create one.

Methods
-------

.. py:method:: Texture.read(alignment: int = 1) -> bytes

    Read the pixel data as bytes into system memory.

    :param int alignment: The byte alignment of the pixels.

.. py:method:: Texture.read_into(buffer: Any, alignment: int = 1, write_offset: int = 0)

    Read the content of the texture into a bytearray or :py:class:`~moderngl.Buffer`.

    The advantage of reading into a :py:class:`~moderngl.Buffer` is that pixel data
    does not need to travel all the way to system memory::

        # Reading pixel data into a bytearray
        data = bytearray(8)
        texture = ctx.texture3d((2, 2, 2), 1)
        texture.read_into(data)

        # Reading pixel data into a buffer
        data = ctx.buffer(reserve=8)
        texture = ctx.texture3d((2, 2, 2), 1)
        texture.read_into(data)

    :param bytearray buffer: The buffer that will receive the pixels.
    :param int alignment: The byte alignment of the pixels.
    :param int write_offset: The write offset.

.. py:method:: Texture.write(data: Any, viewport: tuple, alignment: int = 1)

    Update the content of the texture from byte data or a moderngl :py:class:`~moderngl.Buffer`.

    Examples::

        # Write data from a moderngl Buffer
        data = ctx.buffer(reserve=8)
        texture = ctx.texture3d((2, 2, 2), 1)
        texture.write(data)

        # Write data from bytes
        data = b'\xff\xff\xff\xff\xff\xff\xff\xff'
        texture = ctx.texture3d((2, 2), 1)
        texture.write(data)

    :param bytes data: The pixel data.
    :param tuple viewport: The viewport.
    :param int alignment: The byte alignment of the pixels.

.. py:method:: Texture.build_mipmaps(base: int = 0, max_level: int = 1000) -> None

    Generate mipmaps.

    This also changes the texture filter to ``LINEAR_MIPMAP_LINEAR, LINEAR``
    (Will be removed in ``6.x``)

    :param int base: The base level
    :param int max_level: The maximum levels to generate

.. py:method:: Texture.bind_to_image(unit: int, read: bool = True, write: bool = True, level: int = 0, format: int = 0) -> None

    Bind a texture to an image unit (OpenGL 4.2 required).

    This is used to bind textures to image units for shaders.
    The idea with image load/store is that the user can bind
    one of the images in a Texture to a number of image binding points
    (which are separate from texture image units). Shaders can read
    information from these images and write information to them,
    in ways that they cannot with textures.

    It's important to specify the right access type for the image.
    This can be set with the ``read`` and ``write`` arguments.
    Allowed combinations are:

    - **Read-only**: ``read=True`` and ``write=False``
    - **Write-only**: ``read=False`` and ``write=True``
    - **Read-write**: ``read=True`` and ``write=True``

    ``format`` specifies the format that is to be used when performing
    formatted stores into the image from shaders. ``format`` must be
    compatible with the texture's internal format. **By default the format
    of the texture is passed in. The format parameter is only needed
    when overriding this behavior.**

    Note that we bind the 3D textured layered making the entire texture
    readable and writable. It is possible to bind a specific 2D section
    in the future.

    More information:

    - https://www.khronos.org/opengl/wiki/Image_Load_Store
    - https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBindImageTexture.xhtml

    :param int unit: Specifies the index of the image unit to which to bind the texture
    :param Texture texture: The texture to bind
    :param bool read: Allows the shader to read the image (default: ``True``)
    :param bool write: Allows the shader to write to the image (default: ``True``)
    :param int level: Level of the texture to bind (default: ``0``).
    :param int format: (optional) The OpenGL enum value representing the format (defaults to the texture's format)

.. py:method:: Texture.use(location: int = 0) -> None

    Better to use :py:class:`Sampler` objects and avoid this call on the Texture object directly.

    Bind the texture to a texture unit.

    :param int location: The texture location/unit.

    The location is the texture unit we want to bind the texture.
    This should correspond with the value of the ``sampler2D``
    uniform in the shader because samplers read from the texture
    unit we assign to them::

        # Define what texture unit our two sampler3D uniforms should represent
        program['texture_a'] = 0
        program['texture_b'] = 1
        # Bind textures to the texture units
        first_texture.use(location=0)
        second_texture.use(location=1)

.. py:method:: Texture.get_handle(resident: bool = True) -> int

    Handle for Bindless Textures.

    :param bool resident: Make the texture resident.

    Once a handle is created its parameters cannot be changed.
    Attempting to do so will have no effect. (filter, wrap etc).
    There is no way to undo this immutability.

    Handles cannot be used by shaders until they are resident.
    This method can be called multiple times to move a texture
    in and out of residency::

        >> texture.get_handle(resident=False)
        4294969856
        >> texture.get_handle(resident=True)
        4294969856

    Ths same handle is returned if the handle already exists.

    .. note:: Limitations from the OpenGL wiki

        The amount of storage available for resident images/textures may be less
        than the total storage for textures that is available. As such, you should
        attempt to minimize the time a texture spends being resident. Do not attempt
        to take steps like making textures resident/unresident every frame or something.
        But if you are finished using a texture for some time, make it unresident.

.. py:method:: Texture.release

Attributes
----------

.. py:attribute:: Texture.width
    :type: int

    The width of the texture.

.. py:attribute:: Texture.height
    :type: int

    The height of the texture.

.. py:attribute:: Texture.size
    :type: Tuple[int, int]

    The size of the texture.

.. py:attribute:: Texture.components
    :type: int

    The number of components of the texture.

.. py:attribute:: Texture.samples
    :type: int

    The number of samples set for the texture used in multisampling.

.. py:attribute:: Texture.depth
    :type: bool

    Determines if the texture contains depth values.

.. py:attribute:: Texture.dtype
    :type: str

    Data type.

.. py:attribute:: Texture.swizzle
    :type: str

    The swizzle mask of the texture (Default ``'RGBA'``).

    The swizzle mask change/reorder the ``vec4`` value returned by the ``texture()`` function
    in a GLSL shaders. This is represented by a 4 character string were each
    character can be::

        'R' GL_RED
        'G' GL_GREEN
        'B' GL_BLUE
        'A' GL_ALPHA
        '0' GL_ZERO
        '1' GL_ONE

    Example::

        # Alpha channel will always return 1.0
        texture.swizzle = 'RGB1'

        # Only return the red component. The rest is masked to 0.0
        texture.swizzle = 'R000'

        # Reverse the components
        texture.swizzle = 'ABGR'

.. py:attribute:: Texture.repeat_x

    See :py:class:`Sampler.repeat_x`

.. py:attribute:: Texture.repeat_y

    See :py:class:`Sampler.repeat_y`

.. py:attribute:: Texture.filter

    See :py:class:`Sampler.filter`

.. py:attribute:: Texture.compare_func

    See :py:class:`Sampler.compare_func`

.. py:attribute:: Texture.anisotropy

    See :py:class:`Sampler.anisotropy`

.. py:attribute:: Texture.ctx
    :type: Context

    The context this object belongs to

.. py:attribute:: Texture.glo
    :type: int

    The internal OpenGL object.
    This values is provided for interoperability and debug purposes only.

.. py:attribute:: Texture.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: reference/attribute.rst

Attribute
=========

.. py:class:: Attribute

    Available in :py:meth:`Program.__getitem__`

    Represents a program input attribute.

.. py:attribute:: Attribute.location
    :type: int

    The location of the attribute.
    The result of the glGetAttribLocation.

.. py:attribute:: Attribute.array_length
    :type: int

    If the attribute is an array the array_length is the length of the array otherwise `1`.

.. py:attribute:: Attribute.dimension
    :type: int

    The attribute dimension.

.. py:attribute:: Attribute.shape
    :type: str

    The shape is a single character, representing the scalar type of the attribute.
    It is either ``'i'`` (int), ``'f'`` (float), ``'I'`` (unsigned int), ``'d'`` (double).

.. py:attribute:: Attribute.name
    :type: str

    The attribute name.

    The name will be filtered to have no array syntax on it's end.
    Attribute name without ``'[0]'`` ending if any.

.. py:attribute:: Attribute.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: reference/texture_array.rst

TextureArray
============

.. py:class:: TextureArray

    Returned by :py:meth:`Context.texture_array`

    An Array Texture is a Texture where each mipmap level contains an array of images of the same size.

    Array textures may have Mipmaps, but each mipmap
    in the texture has the same number of levels.

    A TextureArray object cannot be instantiated directly, it requires a context.
    Use :py:meth:`Context.texture_array` to create one.

Methods
-------

.. py:method:: TextureArray.read
.. py:method:: TextureArray.read_into
.. py:method:: TextureArray.write
.. py:method:: TextureArray.bind_to_image
.. py:method:: TextureArray.build_mipmaps
.. py:method:: TextureArray.use
.. py:method:: TextureArray.release
.. py:method:: TextureArray.get_handle

Attributes
----------

.. py:attribute:: TextureArray.repeat_x
.. py:attribute:: TextureArray.repeat_y
.. py:attribute:: TextureArray.filter
.. py:attribute:: TextureArray.swizzle
.. py:attribute:: TextureArray.anisotropy
.. py:attribute:: TextureArray.width
.. py:attribute:: TextureArray.height
.. py:attribute:: TextureArray.layers
.. py:attribute:: TextureArray.size
.. py:attribute:: TextureArray.dtype
.. py:attribute:: TextureArray.components

.. py:attribute:: TextureArray.ctx
    :type: Context

    The context this object belongs to

.. py:attribute:: TextureArray.glo
    :type: int

    The internal OpenGL object.
    This values is provided for interoperability and debug purposes only.

.. py:attribute:: TextureArray.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: reference/conditional_render.rst

ConditionalRender
=================

.. py:class:: ConditionalRender

    Available in :py:attr:`Query.crender`

    This class represents a ConditionalRender object.

    ConditionalRender objects can only be accessed from :py:class:`Query` objects.

Examples
--------

.. code-block:: python

    query = ctx.query(any_samples=True)

    with query:
        vao1.render()

    with query.crender:
        print('This will always get printed')
        vao2.render()  # But this will be rendered only if vao1 has passing samples.


--------------------------------------------------------------------------------


.. File: reference/program.rst

Program
=======

.. py:class:: Program

    Returned by :py:meth:`Context.program`

    A Program object represents fully processed executable code in the OpenGL Shading Language, \
    for one or more Shader stages.

    In ModernGL, a Program object can be assigned to :py:class:`VertexArray` objects.
    The VertexArray object  is capable of binding the Program object once the
    :py:meth:`VertexArray.render` or :py:meth:`VertexArray.transform` is called.

    Program objects has no method called ``use()``, VertexArrays encapsulate this mechanism.

    A Program object cannot be instantiated directly, it requires a context.
    Use :py:meth:`Context.program` to create one.

    Uniform buffers can be bound using :py:meth:`Buffer.bind_to_uniform_block`
    or can be set individually. For more complex binding yielding higher
    performance consider using :py:class:`moderngl.Scope`.

Methods
-------

.. py:method:: Program.get(key: str, default: Any) -> Uniform | UniformBlock | Attribute | Varying

    Returns a Uniform, UniformBlock, Attribute or Varying.

    :param default: This is the value to be returned in case key does not exist.

.. py:method:: Program.__getitem__

    Get a member such as uniforms, uniform blocks, attributes and varyings by name.

    .. code-block:: python

        # Get a uniform
        uniform = program['color']

        # Uniform values can be set on the returned object
        # or the `__setitem__` shortcut can be used.
        program['color'].value = 1.0, 1.0, 1.0, 1.0

        # Still when writing byte data we need to use the `write()` method
        program['color'].write(buffer)

.. py:method:: Program.__setitem__

    Set a value of uniform or uniform block.

    .. code-block:: python

        # Set a vec4 uniform
        uniform['color'] = 1.0, 1.0, 1.0, 1.0

        # Optionally we can store references to a member and set the value directly
        uniform = program['color']
        uniform.value = 1.0, 0.0, 0.0, 0.0

        uniform = program['cameraMatrix']
        uniform.write(camera_matrix)

.. py:method:: Program.__iter__

    Yields the internal members names as strings.

    This includes all members such as uniforms, attributes etc.

    Example::

        # Print member information
        for name in program:
            member = program[name]
            print(name, type(member), member)

    Output::

        vert <class 'moderngl.program_members.attribute.Attribute'> <Attribute: 0>
        vert_color <class 'moderngl.program_members.attribute.Attribute'> <Attribute: 1>
        gl_InstanceID <class 'moderngl.program_members.attribute.Attribute'> <Attribute: -1>
        rotation <class 'moderngl.program_members.uniform.Uniform'> <Uniform: 0>
        scale <class 'moderngl.program_members.uniform.Uniform'> <Uniform: 1>

    We can filter on member type if needed::

        for name in prog:
            member = prog[name]
            if isinstance(member, moderngl.Uniform):
                print('Uniform', name, member)

    or a less verbose version using dict comprehensions::

        uniforms = {name: self.prog[name] for name in self.prog
                    if isinstance(self.prog[name], moderngl.Uniform)}
        print(uniforms)

    Output::

        {'rotation': <Uniform: 0>, 'scale': <Uniform: 1>}

.. py:method:: Program.release() -> None

    Release the ModernGL object.

Attributes
----------

.. py:attribute:: Program.geometry_input
    :type: int

    The geometry input primitive.

    The GeometryShader's input primitive if the GeometryShader exists.
    The geometry input primitive will be used for validation.
    (from ``layout(input_primitive) in;``)

    This can only be ``POINTS``, ``LINES``, ``LINES_ADJACENCY``, ``TRIANGLES``, ``TRIANGLE_ADJACENCY``.

.. py:attribute:: Program.geometry_output
    :type: int

    The geometry output primitive.

    The GeometryShader's output primitive if the GeometryShader exists.
    This can only be ``POINTS``, ``LINE_STRIP`` and ``TRIANGLE_STRIP``
    (from ``layout(output_primitive, max_vertices = vert_count) out;``)

.. py:attribute:: Program.geometry_vertices
    :type: int

    The maximum number of vertices that the geometry shader will output.
    (from ``layout(output_primitive, max_vertices = vert_count) out;``)

.. py:attribute:: Program.is_transform
    :type: int

    If this is a tranform program (no fragment shader).

.. py:attribute:: Program.ctx
    :type: Context

    The context this object belongs to

.. py:attribute:: Program.glo
    :type: int

    The internal OpenGL object.
    This values is provided for interoperability and debug purposes only.

.. py:attribute:: Program.extra
    :type: Any

    User defined data.

Examples
--------

.. rubric:: A simple program designed for rendering

.. code-block:: python
    :linenos:

    my_render_program = ctx.program(
        vertex_shader='''
            #version 330

            in vec2 vert;

            void main() {
                gl_Position = vec4(vert, 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330

            out vec4 color;

            void main() {
                color = vec4(0.3, 0.5, 1.0, 1.0);
            }
        ''',
    )

.. rubric:: A simple program designed for transforming

.. code-block:: python
    :linenos:

    my_transform_program = ctx.program(
        vertex_shader='''
            #version 330

            in vec4 vert;
            out float vert_length;

            void main() {
                vert_length = length(vert);
            }
        ''',
        varyings=['vert_length']
    )

Program Members
---------------

.. toctree::
    :maxdepth: 2

    uniform.rst
    uniform_block.rst
    storage_block.rst
    attribute.rst
    varying.rst


--------------------------------------------------------------------------------


.. File: reference/compute_shader.rst

ComputeShader
=============

.. py:class:: ComputeShader

    Returned by :py:meth:`Context.compute_shader`

    A Compute Shader is a Shader Stage that is used entirely for computing arbitrary information.

    While it can do rendering, it is generally used
    for tasks not directly related to drawing.

    - Compute shaders support uniforms similar to :py:class:`moderngl.Program` objects.
    - Storage buffers can be bound using :py:meth:`Buffer.bind_to_storage_buffer`.
    - Uniform buffers can be bound using :py:meth:`Buffer.bind_to_uniform_block`.
    - Images can be bound using :py:meth:`Texture.bind_to_image`.

Methods
-------

.. py:method:: ComputeShader.run(group_x: int = 1, group_y: int = 1, group_z: int = 1) -> None:

    :param int group_x: Workgroup size x.
    :param int group_y: Workgroup size y.
    :param int group_z: Workgroup size z.

    Run the compute shader.

.. py:method:: run_indirect(self, buffer: Buffer, offset: int = 0) -> None:

    Run the compute shader indirectly from a Buffer object.

    :param Buffer buffer: the buffer containing a single workgroup size at offset.
    :param int offset: the offset into the buffer in bytes.

.. py:method:: ComputeShader.get(key, default)

    Returns a Uniform, UniformBlock or StorageBlock.

    :param default: This is the value to be returned in case key does not exist.

.. py:method:: ComputeShader.__getitem__(key)

    Get a member such as uniforms, uniform blocks and storage blocks.

    .. code-block:: python

        # Get a uniform
        uniform = program['color']

        # Uniform values can be set on the returned object
        # or the `__setitem__` shortcut can be used.
        program['color'].value = 1.0, 1.0, 1.0, 1.0

        # Still when writing byte data we need to use the `write()` method
        program['color'].write(buffer)

        # Set binding for a storage block (if supported)
        program['DataBlock'].binding = 0

.. py:method:: ComputeShader.__setitem__(key, value)

    Set a value of uniform or uniform block.

    .. code-block:: python

        # Set a vec4 uniform
        uniform['color'] = 1.0, 1.0, 1.0, 1.0

        # Optionally we can store references to a member and set the value directly
        uniform = program['color']
        uniform.value = 1.0, 0.0, 0.0, 0.0

        uniform = program['cameraMatrix']
        uniform.write(camera_matrix)

        # Set binding for a storage block (if supported)
        program['DataBlock'].binding = 0

.. py:method:: ComputeShader.__iter__()

    Yields the internal members names as strings.

    Example::

        for member in program:
            obj = program[member]
            print(member, obj)
            if isinstance(obj, moderngl.StorageBlock):
                print('This is a storage block member')

    This includes all members such as uniforms, uniform blocks and storage blocks.

Attributes
----------

.. py:attribute:: ComputeShader.ctx
    :type: Context

    The context this object belongs to

.. py:attribute:: ComputeShader.glo
    :type: int

    The internal OpenGL object.
    This values is provided for interoperability and debug purposes only.

.. py:attribute:: ComputeShader.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: reference/query.rst

Query
=====

.. py:class:: Query

    Returned by :py:meth:`Context.query`

    This class represents a Query object.

Attributes
----------

.. py:attribute:: Query.samples
    :type: int

    The number of samples passed.

.. py:attribute:: Query.primitives
    :type: int

    The number of primitives generated.

.. py:attribute:: Query.elapsed
    :type: int

    The time elapsed in nanoseconds.

.. py:attribute:: Query.crender
    :type: ConditionalRender

.. py:attribute:: Query.ctx
    :type: Context

    The context this object belongs to

.. py:attribute:: Query.extra
    :type: Any

    User defined data.

Examples
--------

.. rubric:: Simple query example

.. code-block:: python
    :linenos:

    import moderngl
    import numpy as np

    ctx = moderngl.create_standalone_context()
    prog = ctx.program(
        vertex_shader='''
            #version 330

            in vec2 in_vert;

            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330

            out vec4 color;

            void main() {
                color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ''',
    )

    vertices = np.array([
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
    ], dtype='f4')

    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')

    fbo = ctx.simple_framebuffer((64, 64))
    fbo.use()

    query = ctx.query(samples=True, time=True)

    with query:
        vao.render()

    print('It took %d nanoseconds' % query.elapsed)
    print('to render %d samples' % query.samples)

.. rubric:: Output

.. code-block:: text

    It took 13529 nanoseconds
    to render 496 samples

.. toctree::
    :maxdepth: 1

    conditional_render.rst


--------------------------------------------------------------------------------


.. File: reference/scope.rst

Scope
=====

.. py:class:: Scope

    Returned by :py:meth:`Context.scope`

    This class represents a Scope object.

    Responsibilities on enter:

    - Set the enable flags.
    - Bind the framebuffer.
    - Assigning textures to texture locations.
    - Assigning buffers to uniform buffers.
    - Assigning buffers to shader storage buffers.

    Responsibilities on exit:

    - Restore the enable flags.
    - Restore the framebuffer.

Methods
-------

.. py:method:: Scope.release

Attributes
----------

.. py:attribute:: Scope.ctx
    :type: Context

    The context this object belongs to

.. py:attribute:: Scope.extra
    :type: Any

    User defined data.

Examples
--------

.. rubric:: Simple scope example

.. code-block:: python

    scope1 = ctx.scope(fbo1, moderngl.BLEND)
    scope2 = ctx.scope(fbo2, moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    with scope1:
        # do some rendering

    with scope2:
        # do some rendering

.. rubric:: Scope for querying

.. code-block:: python

    query = ctx.query(samples=True)
    scope = ctx.scope(ctx.screen, moderngl.DEPTH_TEST | moderngl.RASTERIZER_DISCARD)

    with scope, query:
        # do some rendering

    print(query.samples)

.. rubric:: Understanding what scope objects do

.. code-block:: python

    scope = ctx.scope(
        framebuffer=framebuffer1,
        enable_only=moderngl.BLEND,
        textures=[
            (texture1, 4),
            (texture2, 3),
        ],
        uniform_buffers=[
            (buffer1, 6),
            (buffer2, 5),
        ],
        storage_buffers=[
            (buffer3, 8),
        ],
    )

    # Let's assume we have some state before entering the scope
    some_random_framebuffer.use()
    some_random_texture.use(3)
    some_random_buffer.bind_to_uniform_block(5)
    some_random_buffer.bind_to_storage_buffer(8)
    ctx.enable_only(moderngl.DEPTH_TEST)

    with scope:
        # on __enter__
        #     framebuffer1.use()
        #     ctx.enable_only(moderngl.BLEND)
        #     texture1.use(4)
        #     texture2.use(3)
        #     buffer1.bind_to_uniform_block(6)
        #     buffer2.bind_to_uniform_block(5)
        #     buffer3.bind_to_storage_buffer(8)

        # do some rendering

        # on __exit__
        #     some_random_framebuffer.use()
        #     ctx.enable_only(moderngl.DEPTH_TEST)

    # Originally we had the following, let's see what was changed
    some_random_framebuffer.use()                 # This was restored hurray!
    some_random_texture.use(3)                    # Have to restore it manually.
    some_random_buffer.bind_to_uniform_block(5)   # Have to restore it manually.
    some_random_buffer.bind_to_storage_buffer(8)  # Have to restore it manually.
    ctx.enable_only(moderngl.DEPTH_TEST)          # This was restored too.

    # Scope objects only do as much as necessary.
    # Restoring the framebuffer and enable flags are lowcost operations and
    # without them you could get a hard time debugging the application.


--------------------------------------------------------------------------------


.. File: reference/texture_cube.rst

TextureCube
===========

.. py:class:: TextureCube

    Returned by :py:meth:`Context.texture_cube` and :py:meth:`Context.depth_texture_cube`

    Cubemaps are a texture using the type GL_TEXTURE_CUBE_MAP.

    They are similar to 2D textures in that they have two dimensions.
    However, each mipmap level has 6 faces, with each face having the
    same size as the other faces.

    The width and height of a cubemap must be the same (ie: cubemaps are squares),
    but these sizes need not be powers of two.

    .. Note:: ModernGL enables ``GL_TEXTURE_CUBE_MAP_SEAMLESS`` globally
                to ensure filtering will be done across the cube faces.

    A Texture3D object cannot be instantiated directly, it requires a context.
    Use :py:meth:`Context.texture_cube` to create one.

Methods
-------

.. py:method:: TextureCube.read
.. py:method:: TextureCube.read_into
.. py:method:: TextureCube.write
.. py:method:: TextureCube.bind_to_image
.. py:method:: TextureCube.use
.. py:method:: TextureCube.release
.. py:method:: TextureCube.get_handle

Attributes
----------

.. py:attribute:: TextureCube.size
.. py:attribute:: TextureCube.dtype
.. py:attribute:: TextureCube.components
.. py:attribute:: TextureCube.filter
.. py:attribute:: TextureCube.swizzle
.. py:attribute:: TextureCube.anisotropy

.. py:attribute:: TextureCube.ctx
    :type: Context

    The context this object belongs to

.. py:attribute:: TextureCube.glo
    :type: int

    The internal OpenGL object.
    This values is provided for interoperability and debug purposes only.

.. py:attribute:: TextureCube.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: reference/storage_block.rst

StorageBlock
============

.. py:class:: StorageBlock

    Available in :py:meth:`Program.__getitem__`

    Storage Blocks are OpenGL 4.3+ Program accessible data blocks.
    Compared to UniformBlocks they can be larger in size and also support write operations.
    For less than one page (64KB) read-only data use UniformBlocks.

.. py:attribute:: StorageBlock.binding
    :type: int

    The binding of the Storage block. Same as the value.

.. py:attribute:: StorageBlock.value
    :type: int

    The value of the Storage block. Same as the binding.

.. py:attribute:: StorageBlock.name
    :type: str

    The name of the Storage block.

.. py:attribute:: StorageBlock.index
    :type: int

    The index of the Storage block.

.. py:attribute:: StorageBlock.size
    :type: int

    The size of the Storage block.

.. py:attribute:: StorageBlock.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: reference/vertex_array.rst

VertexArray
===========

.. py:class:: VertexArray

    Returned by :py:meth:`Context.vertex_array`

    A VertexArray object is an OpenGL object that stores all of the state needed to supply vertex data.

    It stores the format of the vertex data
    as well as the Buffer objects providing the vertex data arrays.

    In ModernGL, the VertexArray object also stores a reference
    for a :py:class:`Program` object.

    Compared to OpenGL, :py:class:`VertexArray` also stores a :py:class:`Program` object.

Methods
-------

.. py:method:: VertexArray.render(mode: int | None = None, vertices: int = -1, first: int = 0, instances: int = -1) -> None

    The render primitive (mode) must be the same as the input primitive of the GeometryShader.

    :param int mode: By default :py:data:`TRIANGLES` will be used.
    :param int vertices: The number of vertices to transform.
    :param int first: The index of the first vertex to start with.
    :param int instances: The number of instances.

.. py:method:: VertexArray.render_indirect(buffer: Buffer, mode: int | None = None, count: int = -1, first: int = 0) -> None

    The render primitive (mode) must be the same as the input primitive of the GeometryShader.

    The draw commands are 5 integers: (count, instanceCount, firstIndex, baseVertex, baseInstance).

    :param Buffer buffer: Indirect drawing commands.
    :param int mode: By default :py:data:`TRIANGLES` will be used.
    :param int count: The number of draws.
    :param int first: The index of the first indirect draw command.

.. py:method:: VertexArray.transform(buffer: Buffer | List[Buffer], mode: int | None = None, vertices: int = -1, first: int = 0, instances: int = -1, buffer_offset: int = 0) -> None

    Transform vertices.

    Stores the output in a single buffer.
    The transform primitive (mode) must be the same as
    the input primitive of the GeometryShader.

    :param Buffer buffer: The buffer to store the output.
    :param int mode: By default :py:data:`POINTS` will be used.
    :param int vertices: The number of vertices to transform.
    :param int first: The index of the first vertex to start with.
    :param int instances: The number of instances.
    :param int buffer_offset: Byte offset for the output buffer

.. py:method:: VertexArray.bind(attribute: int, cls: str, buffer: Buffer, fmt: str, offset: int = 0, stride: int = 0, divisor: int = 0, normalize: bool = False)

    Bind individual attributes to buffers.

    :param int location: The attribute location.
    :param str cls: The attribute class. Valid values are ``f``, ``i`` or ``d``.
    :param Buffer buffer: The buffer.
    :param str format: The buffer format.
    :param int offset: The offset.
    :param int stride: The stride.
    :param int divisor: The divisor.
    :param bool normalize: The normalize parameter, if applicable.

.. py:method:: VertexArray.release() -> None

    Release the ModernGL object.

Attributes
----------

.. py:attribute:: VertexArray.mode
    :type: int

    Get or set the default rendering mode.

    This value is used when ``mode`` is not passed in rendering calls.

    Examples::

        vao.mode = moderngl.TRIANGLE_STRIPS

.. py:attribute:: VertexArray.program
    :type: Program

    The program assigned to the VertexArray.
    The program used when rendering or transforming primitives.

.. py:attribute:: VertexArray.index_buffer
    :type: Buffer

    The index buffer if the index_buffer is set, otherwise ``None``.

.. py:attribute:: VertexArray.index_element_size
    :type: int

    The byte size of each element in the index buffer.

.. py:attribute:: VertexArray.scope
    :type: Scope

    The scope to use while rendering.

.. py:attribute:: VertexArray.vertices
    :type: int

    The number of vertices detected.

    This is the minimum of the number of vertices possible per Buffer.
    The size of the index_buffer determines the number of vertices.
    Per instance vertex attributes does not affect this number.

.. py:attribute:: VertexArray.instances
    :type: int

    Get or set the number of instances to render.

.. py:attribute:: VertexArray.ctx
    :type: Context

    The context this object belongs to

.. py:attribute:: VertexArray.glo
    :type: int

    The internal OpenGL object.
    This values is provided for interoperability and debug purposes only.

.. py:attribute:: VertexArray.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: reference/framebuffer.rst

Framebuffer
===========

.. py:class:: Framebuffer

    Returned by :py:meth:`Context.framebuffer`

    A :py:class:`Framebuffer` is a collection of buffers that can be used as the destination for rendering.

    The buffers for Framebuffer objects reference images from either Textures or Renderbuffers.

Methods
-------

.. py:method:: Framebuffer.clear(red: float = 0.0, green: float = 0.0, blue: float = 0.0, alpha: float = 0.0, depth: float = 1.0, viewport=..., color=...) -> None

    Clear the framebuffer.

    If a `viewport` passed in, a scissor test will be used to clear the given viewport.
    This viewport take presence over the framebuffers :py:attr:`~moderngl.Framebuffer.scissor`.
    Clearing can still be done with scissor if no viewport is passed in.

    This method also respects the
    :py:attr:`~moderngl.Framebuffer.color_mask` and
    :py:attr:`~moderngl.Framebuffer.depth_mask`. It can for example be used to only clear
    the depth or color buffer or specific components in the color buffer.

    If the `viewport` is a 2-tuple it will clear the
    ``(0, 0, width, height)`` where ``(width, height)`` is the 2-tuple.

    If the `viewport` is a 4-tuple it will clear the given viewport.

    :param float red: color component.
    :param float green: color component.
    :param float blue: color component.
    :param float alpha: alpha component.
    :param float depth: depth value.
    :param tuple viewport: The viewport.
    :param tuple color: Optional tuple replacing the red, green, blue and alpha arguments

.. py:method:: Framebuffer.read(viewport=..., components: int = 3, attachment: int = 0, alignment: int = 1, dtype: str = 'f1', clamp: bool = False) -> bytes

    Read the content of the framebuffer.

    :param tuple viewport: The viewport.
    :param int components: The number of components to read.
    :param int attachment: The color attachment number. -1 for the depth attachment
    :param int alignment: The byte alignment of the pixels.
    :param str dtype: Data type.
    :param bool clamp: Clamps floating point values to ``[0.0, 1.0]``

    .. code:: python

        # Read the first color attachment's RGBA data
        data = fbo.read(components=4)
        # Read the second color attachment's RGB data
        data = fbo.read(attachment=1)
        # Read the depth attachment
        data = fbo.read(attachment=-1)
        # Read the lower left 10 x 10 pixels from the first color attachment
        data = fbo.read(viewport=(0, 0, 10, 10))

.. py:method:: Framebuffer.read_into(buffer, viewport, components: int = 3, attachment: int = 0, alignment: int = 1, dtype: str = 'f1', write_offset: int = 0) -> None

    Read the content of the framebuffer into a buffer.

    :param bytearray buffer: The buffer that will receive the pixels.
    :param tuple viewport: The viewport.
    :param int components: The number of components to read.
    :param int attachment: The color attachment.
    :param int alignment: The byte alignment of the pixels.
    :param str dtype: Data type.
    :param int write_offset: The write offset.

.. py:method:: Framebuffer.use()

    Bind the framebuffer.

.. py:method:: Framebuffer.release

Attributes
----------

.. py:attribute:: Framebuffer.viewport
    :type: tuple

    Get or set the viewport of the framebuffer.

.. py:attribute:: Framebuffer.scissor
    :type: tuple

    Get or set the scissor box of the framebuffer.

    When scissor testing is enabled fragments outside
    the defined scissor box will be discarded. This
    applies to rendered geometry or :py:meth:`Framebuffer.clear`.

    Setting is value enables scissor testing in the framebuffer.
    Setting the scissor to ``None`` disables scissor testing
    and reverts the scissor box to match the framebuffer size.

    Example::

        # Enable scissor testing
        >>> ctx.scissor = 100, 100, 200, 100
        # Disable scissor testing
        >>> ctx.scissor = None

.. py:attribute:: Framebuffer.color_mask
    :type: tuple

    The color mask of the framebuffer.

    Color masking controls what components in color attachments will be
    affected by fragment write operations.
    This includes rendering geometry and clearing the framebuffer.

    Default value: ``(True, True, True, True)``.

    Examples::

        # Block writing to all color components (rgba) in color attachments
        fbo.color_mask = False, False, False, False

        # Re-enable writing to color attachments
        fbo.color_mask = True, True, True, True

        # Block fragment writes to alpha channel
        fbo.color_mask = True, True, True, False

.. py:attribute:: Framebuffer.depth_mask
    :type: bool

    The depth mask of the framebuffer.

    Depth mask enables or disables write operations to the depth buffer.
    This also applies when clearing the framebuffer.
    If depth testing is enabled fragments will still be culled, but
    the depth buffer will not be updated with new values. This is
    a very useful tool in many rendering techniques.

    Default value: ``True``

.. py:attribute:: Framebuffer.width
    :type: int

    The width of the framebuffer.

    Framebuffers created by a window will only report its initial size.
    It's better get size information from the window itself.

.. py:attribute:: Framebuffer.height
    :type: int

    The height of the framebuffer.

    Framebuffers created by a window will only report its initial size.
    It's better get size information from the window itself.

.. py:attribute:: Framebuffer.size
    :type: Tuple[int, int]

    The size of the framebuffer.

    Framebuffers created by a window will only report its initial size.
    It's better get size information from the window itself.

.. py:attribute:: Framebuffer.samples
    :type: int

    The samples of the framebuffer.

.. py:attribute:: Framebuffer.bits
    :type: int

    The bits of the framebuffer.

.. py:attribute:: Framebuffer.color_attachments
    :type: tuple

.. py:attribute:: Framebuffer.depth_attachment
    :type: tuple

.. py:attribute:: Framebuffer.ctx
    :type: Context

    The context this object belongs to

.. py:attribute:: Framebuffer.glo
    :type: int

    The internal OpenGL object.
    This values is provided for interoperability and debug purposes only.

.. py:attribute:: Framebuffer.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: reference/buffer.rst

Buffer
======

.. py:class:: Buffer

    Returned by :py:meth:`Context.buffer`

    Buffer objects are OpenGL objects that store an array of unformatted memory \
    allocated by the OpenGL context, (data allocated on the GPU).

    These can be used to store vertex data, pixel data retrieved from images
    or the framebuffer, and a variety of other things.

    A Buffer object cannot be instantiated directly, it requires a context.
    Use :py:meth:`Context.buffer` to create one.

    Copy buffer content using :py:meth:`Context.copy_buffer`.

Methods
-------

.. py:method:: Buffer.write(data: Any, *, offset: int = 0) -> None:

    Write the content.

    :param bytes data: The data.
    :param int offset: The offset in bytes.

.. py:method:: Buffer.read(size: int = -1, *, offset: int = 0) -> bytes:

    Read the content.

    :param int size: The size in bytes. Value ``-1`` means all.
    :param int offset: The offset in bytes.

.. py:method:: Buffer.read_into(buffer: Any, size: int = -1, *, offset: int = 0, write_offset: int = 0) -> None:

    Read the content into a buffer.

    :param bytearray buffer: The buffer that will receive the content.
    :param int size: The size in bytes. Value ``-1`` means all.
    :param int offset: The read offset in bytes.
    :param int write_offset: The write offset in bytes.

.. py:method:: Buffer.clear(size: int = -1, *, offset: int = 0, chunk: Any = None) -> None:

    Clear the content.

    :param int size: The size. Value ``-1`` means all.
    :param int offset: The offset.
    :param bytes chunk: The chunk to use repeatedly.

.. py:method:: Buffer.bind_to_uniform_block(binding: int = 0, *, offset: int = 0, size: int = -1) -> None:

    Bind the buffer to a uniform block.

    :param int binding: The uniform block binding.
    :param int offset: The offset.
    :param int size: The size. Value ``-1`` means all.

.. py:method:: Buffer.bind_to_storage_buffer(binding: int = 0, *, offset: int = 0, size: int = -1) -> None:

    Bind the buffer to a shader storage buffer.

    :param int binding: The shader storage binding.
    :param int offset: The offset.
    :param int size: The size. Value ``-1`` means all.

.. py:method:: Buffer.release() -> None:

    Release the ModernGL object

.. py:method:: Buffer.bind(*attribs, layout=None) -> tuple:

    Helper method for binding a buffer in :py:meth:`Context.vertex_array`.

.. py:method:: Buffer.assign(index: int) -> tuple:

    Helper method for assigning a buffer to an index in :py:meth:`Context.scope`.

Attributes
----------

.. py:attribute:: Buffer.size
    :type: int

    The size of the buffer in bytes.

.. py:attribute:: Buffer.dynamic
    :type: bool

    The dynamic flag.

.. py:attribute:: Buffer.ctx
    :type: Context

    The context this object belongs to

.. py:attribute:: Buffer.glo
    :type: int

    The internal OpenGL object.
    This values is provided for interoperability and debug purposes only.

.. py:attribute:: Buffer.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: reference/context.rst

Context
=======

.. py:class:: Context

    Returned by :py:meth:`moderngl.create_context`

    Class exposing OpenGL features.

    ModernGL objects can be created from this class.

Objects
-------

.. py:method:: Context.program(vertex_shader: str, fragment_shader: str, geometry_shader: str, tess_control_shader: str, tess_evaluation_shader: str, varyings: Tuple[str, ...], fragment_outputs: Dict[str, int], varyings_capture_mode: str = 'interleaved') -> Program

    Create a :py:class:`Program` object.

    The ``varyings`` are only used when a transform program is created
    to specify the names of the output varyings to capture in the output buffer.

    ``fragment_outputs`` can be used to programmatically map named fragment
    shader outputs to a framebuffer attachment numbers. This can also be done
    by using ``layout(location=N)`` in the fragment shader.

    :param str vertex_shader: The vertex shader source.
    :param str fragment_shader: The fragment shader source.
    :param str geometry_shader: The geometry shader source.
    :param str tess_control_shader: The tessellation control shader source.
    :param str tess_evaluation_shader: The tessellation evaluation shader source.
    :param list varyings: A list of varyings.
    :param dict fragment_outputs: A dictionary of fragment outputs.

.. py:method:: Context.buffer(data = None, reserve: int = 0, dynamic: bool = False) -> Buffer

    Returns a new :py:class:`Buffer` object.

    The `data` can be anything supporting the buffer interface.

    The `data` and `reserve` parameters are mutually exclusive.

    :param bytes data: Content of the new buffer.
    :param int reserve: The number of bytes to reserve.
    :param bool dynamic: Treat buffer as dynamic.

.. py:method:: Context.vertex_array(program: Program, content: list, index_buffer: Buffer = None, index_element_size: int = 4, mode: int = ...) -> VertexArray

    Returns a new :py:class:`VertexArray` object.

    A VertexArray describes how buffers are read by a shader program.
    The content is a list of tuples containing a buffer, a format string and any number of attribute names.
    Attribute names are defined by the user in the Vertex Shader program stage.

    The default `mode` is :py:attr:`~Context.TRIANGLES`.

    :param Program program: The program used when rendering
    :param list content: A list of (buffer, format, attributes). See :ref:`buffer-format-label`.
    :param Buffer index_buffer: An index buffer (optional)
    :param int index_element_size: byte size of each index element, 1, 2 or 4.
    :param bool skip_errors: Ignore errors during creation
    :param int mode: The default draw mode (for example: ``TRIANGLES``)

    Examples::

        # Empty vertext array (no attribute input)
        vao = ctx.vertex_array(program)

        # Multiple buffers
        vao = ctx.vertex_array(program, [
            (buffer1, '3f', 'in_position'),
            (buffer2, '3f', 'in_normal'),
        ])
        vao = ctx.vertex_array(
            program,
            [
                (buffer1, '3f', 'in_position'),
                (buffer2, '3f', 'in_normal'),
            ],
            index_buffer=ibo,
            index_element_size=2,  # 16 bit / 'u2' index buffer
        )

    Backward Compatible Version::

        # Simple version with a single buffer
        vao = ctx.vertex_array(program, buffer, 'in_position', 'in_normal')
        vao = ctx.vertex_array(program, buffer, 'in_position', 'in_normal', index_buffer=ibo)

.. py:method:: Context.simple_vertex_array(...)

    Deprecated, use :py:meth:`Context.vertex_array` instead.

.. py:method:: Context.texture(size: Tuple[int, int], components: int, data: Any = None, samples: int = 0, alignment: int = 1, dtype: str = 'f1') -> Texture

    Returns a new :py:class:`Texture` object.

    A Texture is a 2D image that can be used for sampler2D uniforms or as render targets if framebuffers.

    :param tuple size: The width and height of the texture.
    :param int components: The number of components 1, 2, 3 or 4.
    :param bytes data: Content of the texture.
    :param int samples: The number of samples. Value 0 means no multisample format.
    :param int alignment: The byte alignment 1, 2, 4 or 8.
    :param str dtype: Data type.
    :param int internal_format: Override the internalformat of the texture (IF needed)

    Example::

        from PIL import Image

        img = Image.open(...).convert('RGBA')
        texture = ctx.texture(img.size, components=4, data=img.tobytes())

        # float texture
        texture = ctx.texture((64, 64), components=..., dtype='f4')

        # integer texture
        texture = ctx.texture((64, 64), components=..., dtype='i4')

    .. Note:: Do not play with ``internal_format`` unless you know exactly
                    you are doing. This is an override to support sRGB and
                    compressed textures if needed.

.. py:method:: Context.framebuffer(color_attachments: List[Texture], depth_attachment: Texture = None) -> Framebuffer

    Returns a new :py:class:`Framebuffer` object.

    A Framebuffer is a collection of images that can be used as render targets.
    The images of the Framebuffer object can be either Textures or Renderbuffers.

    :param list color_attachments: A list of :py:class:`Texture` or :py:class:`Renderbuffer` objects.
    :param Texture depth_attachment: The depth attachment.

.. py:method:: Context.sampler(repeat_x: bool, repeat_y: bool, repeat_z: bool, filter: tuple, anisotropy: float, compare_func: str, border_color: tuple, min_lod: float, max_lod: float, texture: Texture) -> Sampler

    Returns a new :py:class:`Sampler` object.

    Samplers bind Textures to uniform samplers within a Program object.
    Binding a Sampler object also binds the texture object attached to it.

    :param bool repeat_x: Repeat texture on x
    :param bool repeat_y: Repeat texture on y
    :param bool repeat_z: Repeat texture on z
    :param tuple filter: The min and max filter
    :param float anisotropy: Number of samples for anisotropic filtering. Any value greater than 1.0 counts as a use of anisotropic filtering
    :param str compare_func: Compare function for depth textures
    :param tuple border_color: The (r, g, b, a) color for the texture border. When this value is set the ``repeat_`` values are overridden setting the texture wrap to return the border color when outside ``[0, 1]`` range.
    :param float min_lod: Minimum level-of-detail parameter (Default ``-1000.0``). This floating-point value limits the selection of highest resolution mipmap (lowest mipmap level)
    :param float max_lod: Minimum level-of-detail parameter (Default ``1000.0``). This floating-point value limits the selection of the lowest resolution mipmap (highest mipmap level)
    :param Texture texture: The texture for this sampler

.. py:method:: Context.depth_texture(size: Tuple[int, int], data: Any = None, samples: int = 0, alignment: int = 4) -> Texture

    Returns a new :py:class:`Texture` object.

    A depth texture can be used for sampler2D and sampler2DShadow uniforms and as a depth attachment for framebuffers.

    :param tuple size: The width and height of the texture.
    :param bytes data: Content of the texture.
    :param int samples: The number of samples. Value 0 means no multisample format.
    :param int alignment: The byte alignment 1, 2, 4 or 8.

.. py:method:: Context.texture3d(size: Tuple[int, int, int], components: int, data: Any = None, alignment: int = 1, dtype: str = 'f1') -> Texture3D

    Returns a new :py:class:`Texture3D` object.

    :param tuple size: The width, height and depth of the texture.
    :param int components: The number of components 1, 2, 3 or 4.
    :param bytes data: Content of the texture.
    :param int alignment: The byte alignment 1, 2, 4 or 8.
    :param str dtype: Data type.

.. py:method:: Context.texture_array(size: Tuple[int, int, int], components: int, data: Any = None, *, alignment: int = 1, dtype: str = 'f1') -> TextureArray

    Returns a new :py:class:`TextureArray` object.

    :param tuple size: The ``(width, height, layers)`` of the texture.
    :param int components: The number of components 1, 2, 3 or 4.
    :param bytes data: Content of the texture. The size must be ``(width, height * layers)`` so each layer is stacked vertically.
    :param int alignment: The byte alignment 1, 2, 4 or 8.
    :param str dtype: Data type.

.. py:method:: Context.texture_cube(size: Tuple[int, int], components: int, data: Any = None, alignment: int = 1, dtype: str = 'f1') -> TextureCube

    Returns a new :py:class:`TextureCube` object.

    Note that the width and height of the cubemap must be the same.

    :param tuple size: The width, height of the texture. Each side of the cube will have this size.
    :param int components: The number of components 1, 2, 3 or 4.
    :param bytes data: Content of the texture. The data should be have the following ordering: positive_x, negative_x, positive_y, negative_y, positive_z, negative_z
    :param int alignment: The byte alignment 1, 2, 4 or 8.
    :param str dtype: Data type.
    :param int internal_format: Override the internalformat of the texture (IF needed)

.. py:method:: Context.depth_texture_cube(size: Tuple[int, int], data: Optional[Any] = None, alignment: int = 4) -> TextureCube

    Returns a new :py:class:`TextureCube` object.

    :param tuple size: The width and height of the texture.
    :param bytes data: Content of the texture.
    :param int alignment: The byte alignment 1, 2, 4 or 8.

.. py:method:: Context.simple_framebuffer(...)

    Deprecated, use :py:meth:`Context.framebuffer` instead.

.. py:method:: Context.renderbuffer(size: Tuple[int, int], components: int = 4, samples: int = 0, dtype: str = 'f1') -> Renderbuffer

    Returns a new :py:class:`Renderbuffer` object.

    Similar to textures, renderbuffers can be attached to framebuffers as render targets, but they cannot be sampled as textures.

    :param tuple size: The width and height of the renderbuffer.
    :param int components: The number of components 1, 2, 3 or 4.
    :param int samples: The number of samples. Value 0 means no multisample format.
    :param str dtype: Data type.

.. py:method:: Context.depth_renderbuffer(size: Tuple[int, int], samples: int = 0) -> Renderbuffer

    Returns a new :py:class:`Renderbuffer` object.

    :param tuple size: The width and height of the renderbuffer.
    :param int samples: The number of samples. Value 0 means no multisample format.

.. py:method:: Context.scope(framebuffer, enable_only, textures, uniform_buffers, storage_buffers, samplers)

    Returns a new :py:class:`Scope` object.

    Scope objects can be attached to VertexArray objects to minimize the possibility of rendering within the wrong scope.
    VertexArrays with an attached scope always have the scope settings at render time.

    :param Framebuffer framebuffer: The framebuffer to use when entering.
    :param int enable_only: The enable_only flags to set when entering.
    :param tuple textures: List of (texture, binding) tuples.
    :param tuple uniform_buffers: Tuple of (buffer, binding) tuples.
    :param tuple storage_buffers: Tuple of (buffer, binding) tuples.
    :param tuple samplers: Tuple of sampler bindings

.. py:method:: Context.query(samples: bool, any_samples: bool, time: bool, primitives: bool) -> Query

    Returns a new :py:class:`Query` object.

    :param bool samples: Query ``GL_SAMPLES_PASSED`` or not.
    :param bool any_samples: Query ``GL_ANY_SAMPLES_PASSED`` or not.
    :param bool time: Query ``GL_TIME_ELAPSED`` or not.
    :param bool primitives: Query ``GL_PRIMITIVES_GENERATED`` or not.

.. py:method:: Context.compute_shader(...)

    A :py:class:`ComputeShader` is a Shader Stage that is used entirely \
    for computing arbitrary information. While it can do rendering, it \
    is generally used for tasks not directly related to drawing.

    :param str source: The source of the compute shader.

External Objects
----------------

External objects are only useful for interoperability with other libraries.

.. py:method:: Context.external_buffer(glo: int, size: int, dynamic: bool) -> Buffer

    TBD

.. py:method:: Context.external_texture(glo: int, size: Tuple[int, int], components: int, samples: int, dtype: str) -> Texture

    Returns a new :py:class:`Texture` object from an existing OpenGL texture object.

    The content of the texture is referenced and it is not copied.

    :param int glo: External OpenGL texture object.
    :param tuple size: The width and height of the texture.
    :param int components: The number of components 1, 2, 3 or 4.
    :param int samples: The number of samples. Value 0 means no multisample format.
    :param str dtype: Data type.

Methods
-------

.. py:method:: Context.clear

    Clear the bound framebuffer.

    If a `viewport` passed in, a scissor test will be used to clear the given viewport.
    This viewport take prescense over the framebuffers :py:attr:`~moderngl.Framebuffer.scissor`.
    Clearing can still be done with scissor if no viewport is passed in.

    This method also respects the
    :py:attr:`~moderngl.Framebuffer.color_mask` and
    :py:attr:`~moderngl.Framebuffer.depth_mask`. It can for example be used to only clear
    the depth or color buffer or specific components in the color buffer.

    If the `viewport` is a 2-tuple it will clear the
    ``(0, 0, width, height)`` where ``(width, height)`` is the 2-tuple.

    If the `viewport` is a 4-tuple it will clear the given viewport.

    Args:
        red (float): color component.
        green (float): color component.
        blue (float): color component.
        alpha (float): alpha component.
        depth (float): depth value.

    Keyword Args:
        viewport (tuple): The viewport.
        color (tuple): Optional rgba color tuple

.. py:method:: Context.enable_only

    Clears all existing flags applying new ones.

    Note that the enum values defined in moderngl
    are not the same as the ones in opengl.
    These are defined as bit flags so we can logical
    `or` them together.

    Available flags:

    - :py:data:`moderngl.NOTHING`
    - :py:data:`moderngl.BLEND`
    - :py:data:`moderngl.DEPTH_TEST`
    - :py:data:`moderngl.CULL_FACE`
    - :py:data:`moderngl.RASTERIZER_DISCARD`
    - :py:data:`moderngl.PROGRAM_POINT_SIZE`

    Examples::

        # Disable all flags
        ctx.enable_only(moderngl.NOTHING)

        # Ensure only depth testing and face culling is enabled
        ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    Args:
        flags (EnableFlag): The flags to enable

.. py:method:: Context.enable

    Enable flags.

    Note that the enum values defined in moderngl
    are not the same as the ones in opengl.
    These are defined as bit flags so we can logical
    `or` them together.

    For valid flags, please see :py:meth:`enable_only`.

    Examples::

        # Enable a single flag
        ctx.enable(moderngl.DEPTH_TEST)

        # Enable multiple flags
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)

    Args:
        flag (int): The flags to enable.

.. py:method:: Context.disable

    Disable flags.

    For valid flags, please see :py:meth:`enable_only`.

    Examples::

        # Only disable depth testing
        ctx.disable(moderngl.DEPTH_TEST)

        # Disable depth testing and face culling
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    Args:
        flag (int): The flags to disable.

.. py:method:: Context.enable_direct

    Gives direct access to ``glEnable`` so unsupported capabilities in ModernGL can be enabled.

    Do not use this to set already supported context flags.

    Example::

        # Enum value from the opengl registry
        GL_CONSERVATIVE_RASTERIZATION_NV = 0x9346
        ctx.enable_direct(GL_CONSERVATIVE_RASTERIZATION_NV)

.. py:method:: Context.disable_direct

    Gives direct access to ``glDisable`` so unsupported capabilities in ModernGL can be disabled.

    Do not use this to set already supported context flags.

    Example::

        # Enum value from the opengl registry
        GL_CONSERVATIVE_RASTERIZATION_NV = 0x9346
        ctx.disable_direct(GL_CONSERVATIVE_RASTERIZATION_NV)

.. py:method:: Context.finish

    Wait for all drawing commands to finish.

.. py:method:: Context.clear_samplers

    Unbinds samplers from texture units.

    Sampler bindings do clear automatically between every frame,
    but lingering samplers can still be a source of weird bugs during
    the frame rendering. This methods provides a fairly brute force
    and efficient way to ensure texture units are clear.

    :param int start: The texture unit index to start the clearing samplers
    :param int stop: The texture unit index to stop clearing samplers

    Example::

        # Clear texture unit 0, 1, 2, 3, 4
        ctx.clear_samplers(start=0, end=5)

        # Clear texture unit 4, 5, 6, 7
        ctx.clear_samplers(start=4, end=8)

.. py:method:: Context.copy_buffer

    Copy buffer content.

    Args:
        dst (Buffer): The destination buffer.
        src (Buffer): The source buffer.
        size (int): The number of bytes to copy.

    Keyword Args:
        read_offset (int): The read offset.
        write_offset (int): The write offset.

.. py:method:: Context.copy_framebuffer

    Copy framebuffer content.

    Use this method to:

        - blit framebuffers.
        - copy framebuffer content into a texture.
        - downsample framebuffers. (it will allow to read the framebuffer's content)
        - downsample a framebuffer directly to a texture.

    Args:
        dst (Framebuffer or Texture): Destination framebuffer or texture.
        src (Framebuffer): Source framebuffer.

.. py:method:: Context.detect_framebuffer

    Detect a framebuffer.

    This is already done when creating a context,
    but if the underlying window library for some changes the default framebuffer
    during the lifetime of the application this might be necessary.

    Args:
        glo (int): Framebuffer object.

.. py:method:: Context.memory_barrier

    Applying a memory barrier.

    The memory barrier is needed in particular to correctly change buffers or textures
    between each shader. If the same buffer is changed in two shaders,
    it can cause an effect like 'depth fighting' on a buffer or texture.

    The method should be used between :py:class:`Program` -s, between :py:class:`ComputeShader` -s,
    and between :py:class:`Program` -s and :py:class:`ComputeShader` -s.

    Keyword Args:
        barriers (int): Affected barriers, default moderngl.ALL_BARRIER_BITS.
        by_region (bool): Memory barrier mode by region. More read on https://registry.khronos.org/OpenGL-Refpages/gl4/html/glMemoryBarrier.xhtml

.. py:method:: Context.gc() -> int

    Deletes OpenGL objects.
    Returns the number of objects deleted.

    This method must be called to garbage collect
    OpenGL resources when ``gc_mode`` is ``'context_gc'```.

    Calling this method with any other ``gc_mode`` configuration
    has no effect and is perfectly safe.

.. py:method:: Context.release

Attributes
----------

.. py:attribute:: Context.gc_mode
    :type: str

    The garbage collection mode.

    The default mode is ``None`` meaning no automatic
    garbage collection is done. Other modes are ``auto``
    and ``context_gc``. Please see documentation for
    the appropriate configuration.

    Examples::

        # Disable automatic garbage collection.
        # Each objects needs to be explicitly released.
        ctx.gc_mode = None

        # Collect all dead objects in the context and
        # release them by calling Context.gc()
        ctx.gc_mode = 'context_gc'
        ctx.gc()  # Deletes the collected objects

        # Enable automatic garbage collection like
        # we normally expect in python.
        ctx.gc_mode = 'auto'

.. py:attribute:: Context.objects
    :type: deque

    Moderngl objects scheduled for deletion.

    These are deleted when calling :py:meth:`Context.gc`.

.. py:attribute:: Context.line_width
    :type: float

    Set the default line width.

    .. Warning:: A line width other than 1.0 is not guaranteed to work
                    across different OpenGL implementations. For wide
                    lines you should be using geometry shaders.

.. py:attribute:: Context.point_size
    :type: float

    Set/get the point size.

    Point size changes the pixel size of rendered points. The min and max values
    are limited by ``POINT_SIZE_RANGE``.
    This value usually at least ``(1, 100)``, but this depends on the drivers/vendors.

    If variable point size is needed you can enable ``PROGRAM_POINT_SIZE``
    and write to ``gl_PointSize`` in the vertex or geometry shader.

    .. Note::

        Using a geometry shader to create triangle strips from points is often a safer
        way to render large points since you don't have have any size restrictions.

.. py:attribute:: Context.depth_func
    :type: str

    Set the default depth func.

    Example::

        ctx.depth_func = '<='  # GL_LEQUAL
        ctx.depth_func = '<'   # GL_LESS
        ctx.depth_func = '>='  # GL_GEQUAL
        ctx.depth_func = '>'   # GL_GREATER
        ctx.depth_func = '=='  # GL_EQUAL
        ctx.depth_func = '!='  # GL_NOTEQUAL
        ctx.depth_func = '0'   # GL_NEVER
        ctx.depth_func = '1'   # GL_ALWAYS

.. py:attribute:: Context.depth_clamp_range
    :type: Tuple[float, float]

    Setting up depth clamp range (write only, by default ``None``).

    ``ctx.depth_clamp_range`` offers uniform use of GL_DEPTH_CLAMP and glDepthRange.

    ``GL_DEPTH_CLAMP`` is needed to disable clipping of fragments outside
    near limit of projection matrix.
    For example, this will allow you to draw between 0 and 1 in the Z (depth) coordinate,
    even if ``near`` is set to 0.5 in the projection matrix.

    .. note::

        All fragments outside the ``near`` of the projection matrix will have a depth of ``near``.

    See https://www.khronos.org/opengl/wiki/Vertex_Post-Processing#Depth_clamping for more info.

    ``glDepthRange(nearVal, farVal)`` is needed to specify mapping of depth values from normalized device coordinates to window coordinates.
    See https://registry.khronos.org/OpenGL-Refpages/gl4/html/glDepthRange.xhtml for more info.

    Example::

        # For glDisable(GL_DEPTH_CLAMP) and glDepthRange(0, 1)
        ctx.depth_clamp_range = None

        # For glEnable(GL_DEPTH_CLAMP) and glDepthRange(near, far)
        ctx.depth_clamp_range = (near, far)

.. py:attribute:: Context.blend_func
    :type: tuple

    Set the blend func (write only).

    Blend func can be set for rgb and alpha separately if needed.

    Supported blend functions are::

        moderngl.ZERO
        moderngl.ONE
        moderngl.SRC_COLOR
        moderngl.ONE_MINUS_SRC_COLOR
        moderngl.DST_COLOR
        moderngl.ONE_MINUS_DST_COLOR
        moderngl.SRC_ALPHA
        moderngl.ONE_MINUS_SRC_ALPHA
        moderngl.DST_ALPHA
        moderngl.ONE_MINUS_DST_ALPHA

        # Shortcuts
        moderngl.DEFAULT_BLENDING     # (SRC_ALPHA, ONE_MINUS_SRC_ALPHA)
        moderngl.ADDITIVE_BLENDING    # (ONE, ONE)
        moderngl.PREMULTIPLIED_ALPHA  # (SRC_ALPHA, ONE)

    Example::

        # For both rgb and alpha
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Separate for rgb and alpha
        ctx.blend_func = (
            moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
            moderngl.ONE, moderngl.ONE
        )

.. py:attribute:: Context.blend_equation
    :type: tuple

    Set the blend equation (write only).

    Blend equations specify how source and destination colors are combined
    in blending operations. By default ``FUNC_ADD`` is used.

    Blend equation can be set for rgb and alpha separately if needed.

    Supported functions are::

        moderngl.FUNC_ADD               # source + destination
        moderngl.FUNC_SUBTRACT          # source - destination
        moderngl.FUNC_REVERSE_SUBTRACT  # destination - source
        moderngl.MIN                    # Minimum of source and destination
        moderngl.MAX                    # Maximum of source and destination

    Example::

        # For both rgb and alpha channel
        ctx.blend_equation = moderngl.FUNC_ADD

        # Separate for rgb and alpha channel
        ctx.blend_equation = moderngl.FUNC_ADD, moderngl.MAX

.. py:attribute:: Context.multisample
    :type: bool

    Enable/disable multisample mode (``GL_MULTISAMPLE``).

    This property is write only.

    Example::

        # Enable
        ctx.multisample = True
        # Disable
        ctx.multisample = False

.. py:attribute:: Context.viewport
    :type: tuple

    Get or set the viewport of the active framebuffer.

    Example::

        >>> ctx.viewport
        (0, 0, 1280, 720)
        >>> ctx.viewport = (0, 0, 640, 360)
        >>> ctx.viewport
        (0, 0, 640, 360)

    If no framebuffer is bound ``(0, 0, 0, 0)`` will be returned.

.. py:attribute:: Context.scissor
    :type: tuple

    Get or set the scissor box for the active framebuffer.

    When scissor testing is enabled fragments outside
    the defined scissor box will be discarded. This
    applies to rendered geometry or :py:meth:`Context.clear`.

    Setting is value enables scissor testing in the framebuffer.
    Setting the scissor to ``None`` disables scissor testing
    and reverts the scissor box to match the framebuffer size.

    Example::

        # Enable scissor testing
        >>> ctx.scissor = 100, 100, 200, 100
        # Disable scissor testing
        >>> ctx.scissor = None

    If no framebuffer is bound ``(0, 0, 0, 0)`` will be returned.

.. py:attribute:: Context.version_code
    :type: int



.. py:attribute:: Context.screen
    :type: Framebuffer

    A Framebuffer instance representing the screen.

    Normally set when creating a context with ``create_context()`` attaching to
    an existing context. This is the special system framebuffer
    represented by framebuffer ``id=0``.

    When creating a standalone context this property is not set since
    there are no default framebuffer.

.. py:attribute:: Context.fbo
    :type: Framebuffer



.. py:attribute:: Context.front_face
    :type: str

    The front_face. Acceptable values are ``'ccw'`` (default) or ``'cw'``.

    Face culling must be enabled for this to have any effect:
    ``ctx.enable(moderngl.CULL_FACE)``.

    Example::

        # Triangles winded counter-clockwise considered front facing
        ctx.front_face = 'ccw'
        # Triangles winded clockwise considered front facing
        ctx.front_face = 'cw'

.. py:attribute:: Context.cull_face
    :type: str

    The face side to cull. Acceptable values are ``'back'`` (default) ``'front'`` or ``'front_and_back'``.

    This is similar to :py:meth:`Context.front_face`

    Face culling must be enabled for this to have any effect:
    ``ctx.enable(moderngl.CULL_FACE)``.

    Example::

        ctx.cull_face = 'front'
        ctx.cull_face = 'back'
        ctx.cull_face = 'front_and_back'

.. py:attribute:: Context.wireframe
    :type: bool

    Wireframe settings for debugging.

.. py:attribute:: Context.max_samples
    :type: int

    The maximum supported number of samples for multisampling.

.. py:attribute:: Context.max_integer_samples
    :type: int

    The max integer samples.

.. py:attribute:: Context.max_texture_units
    :type: int

    The max texture units.

.. py:attribute:: Context.max_anisotropy
    :type: float

    The maximum value supported for anisotropic filtering.

.. py:attribute:: Context.default_texture_unit
    :type: int

    The default texture unit.

.. py:attribute:: Context.patch_vertices
    :type: int

    The number of vertices that will be used to make up a single patch primitive.

.. py:attribute:: Context.provoking_vertex
    :type: int

    Specifies the vertex to be used as the source of data for flat shaded varyings.

    Flatshading a vertex shader varying output (ie. ``flat out vec3 pos``) means to assign
    all vetices of the primitive the same value for that output. The vertex from which
    these values is derived is known as the provoking vertex.

    It can be configured to be the first or the last vertex.

    This property is write only.

    Example::

        # Use first vertex
        ctx.provoking_vertex = moderngl.FIRST_VERTEX_CONVENTION

        # Use last vertex
        ctx.provoking_vertex = moderngl.LAST_VERTEX_CONVENTION

.. py:attribute:: Context.polygon_offset
    :type: tuple

    Get or set the current polygon offset.

    The tuple values represents two float values: ``unit`` and a ``factor``::

        ctx.polygon_offset = unit, factor

    When drawing polygons, lines or points directly on top of
    exiting geometry the result is often not visually pleasant.
    We can experience z-fighting or partially fading fragments
    due to different primitives not being rasterized in the exact
    same way or simply depth buffer precision issues.

    For example when visualizing polygons drawing a wireframe
    version on top of the original mesh, these issues are
    immediately apparent. Applying decals to surfaces is
    another common example.

    The official documentation states the following::

        When enabled, the depth value of each fragment is added
        to a calculated offset value. The offset is added before
        the depth test is performed and before the depth value
        is written into the depth buffer. The offset value o is calculated by:
        o = m * factor + r * units
        where m is the maximum depth slope of the polygon and r is the smallest
        value guaranteed to produce a resolvable difference in window coordinate
        depth values. The value r is an implementation-specific int.

    In simpler terms: We use polygon offset to either add a positive offset to
    the geometry (push it away from you) or a negative offset to geometry
    (pull it towards you).

    * ``units`` is a int offset to depth and will do the job alone
        if we are working with geometry parallel to the near/far plane.
    * The ``factor`` helps you handle sloped geometry (not parallel to near/far plane).

    In most cases you can get away with ``[-1.0, 1.0]`` for both factor and units,
    but definitely play around with the values. When both values are set to ``0``
    polygon offset is disabled internally.

    To just get started with something you can try::

        # Either push the geomtry away or pull it towards you
        # with support for handling small to medium sloped geometry
        ctx.polygon_offset = 1.0, 1.0
        ctx.polygon_offset = -1.0, -1.0

        # Disable polygon offset
        ctx.polygon_offset = 0, 0

.. py:attribute:: Context.error
    :type: str

    The result of ``glGetError()`` but human readable.

    This values is provided for debug purposes only and is likely to
    reduce performace when used in a draw loop.

.. py:attribute:: Context.extensions
    :type: Set[str]

    The extensions supported by the context.

    All extensions names have a ``GL_`` prefix, so if the spec refers to ``ARB_compute_shader``
    we need to look for ``GL_ARB_compute_shader``::

        # If compute shaders are supported ...
        >> 'GL_ARB_compute_shader' in ctx.extensions
        True

    Example data::

        {
            'GL_ARB_multi_bind',
            'GL_ARB_shader_objects',
            'GL_ARB_half_float_vertex',
            'GL_ARB_map_buffer_alignment',
            'GL_ARB_arrays_of_arrays',
            'GL_ARB_pipeline_statistics_query',
            'GL_ARB_provoking_vertex',
            'GL_ARB_gpu_shader5',
            'GL_ARB_uniform_buffer_object',
            'GL_EXT_blend_equation_separate',
            'GL_ARB_tessellation_shader',
            'GL_ARB_multi_draw_indirect',
            'GL_ARB_multisample',
            .. etc ..
        }

.. py:attribute:: Context.info
    :type: Dict[str, Any]

    OpenGL Limits and information about the context.

    Example::

        # The maximum width and height of a texture
        >> ctx.info['GL_MAX_TEXTURE_SIZE']
        16384

        # Vendor and renderer
        >> ctx.info['GL_VENDOR']
        NVIDIA Corporation
        >> ctx.info['GL_RENDERER']
        NVIDIA GeForce GT 650M OpenGL Engine

    Example data::

        {
            'GL_VENDOR': 'NVIDIA Corporation',
            'GL_RENDERER': 'NVIDIA GeForce GT 650M OpenGL Engine',
            'GL_VERSION': '4.1 NVIDIA-10.32.0 355.11.10.10.40.102',
            'GL_POINT_SIZE_RANGE': (1.0, 2047.0),
            'GL_SMOOTH_LINE_WIDTH_RANGE': (0.5, 1.0),
            'GL_ALIASED_LINE_WIDTH_RANGE': (1.0, 1.0),
            'GL_POINT_FADE_THRESHOLD_SIZE': 1.0,
            'GL_POINT_SIZE_GRANULARITY': 0.125,
            'GL_SMOOTH_LINE_WIDTH_GRANULARITY': 0.125,
            'GL_MIN_PROGRAM_TEXEL_OFFSET': -8.0,
            'GL_MAX_PROGRAM_TEXEL_OFFSET': 7.0,
            'GL_MINOR_VERSION': 1,
            'GL_MAJOR_VERSION': 4,
            'GL_SAMPLE_BUFFERS': 0,
            'GL_SUBPIXEL_BITS': 8,
            'GL_CONTEXT_PROFILE_MASK': 1,
            'GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT': 256,
            'GL_DOUBLEBUFFER': False,
            'GL_STEREO': False,
            'GL_MAX_VIEWPORT_DIMS': (16384, 16384),
            'GL_MAX_3D_TEXTURE_SIZE': 2048,
            'GL_MAX_ARRAY_TEXTURE_LAYERS': 2048,
            'GL_MAX_CLIP_DISTANCES': 8,
            'GL_MAX_COLOR_ATTACHMENTS': 8,
            'GL_MAX_COLOR_TEXTURE_SAMPLES': 8,
            'GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS': 233472,
            'GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS': 231424,
            'GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS': 80,
            'GL_MAX_COMBINED_UNIFORM_BLOCKS': 70,
            'GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS': 233472,
            'GL_MAX_CUBE_MAP_TEXTURE_SIZE': 16384,
            'GL_MAX_DEPTH_TEXTURE_SAMPLES': 8,
            'GL_MAX_DRAW_BUFFERS': 8,
            'GL_MAX_DUAL_SOURCE_DRAW_BUFFERS': 1,
            'GL_MAX_ELEMENTS_INDICES': 150000,
            'GL_MAX_ELEMENTS_VERTICES': 1048575,
            'GL_MAX_FRAGMENT_INPUT_COMPONENTS': 128,
            'GL_MAX_FRAGMENT_UNIFORM_COMPONENTS': 4096,
            'GL_MAX_FRAGMENT_UNIFORM_VECTORS': 1024,
            'GL_MAX_FRAGMENT_UNIFORM_BLOCKS': 14,
            'GL_MAX_GEOMETRY_INPUT_COMPONENTS': 128,
            'GL_MAX_GEOMETRY_OUTPUT_COMPONENTS': 128,
            'GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS': 16,
            'GL_MAX_GEOMETRY_UNIFORM_BLOCKS': 14,
            'GL_MAX_GEOMETRY_UNIFORM_COMPONENTS': 2048,
            'GL_MAX_INTEGER_SAMPLES': 1,
            'GL_MAX_SAMPLES': 8,
            'GL_MAX_RECTANGLE_TEXTURE_SIZE': 16384,
            'GL_MAX_RENDERBUFFER_SIZE': 16384,
            'GL_MAX_SAMPLE_MASK_WORDS': 1,
            'GL_MAX_SERVER_WAIT_TIMEOUT': -1,
            'GL_MAX_TEXTURE_BUFFER_SIZE': 134217728,
            'GL_MAX_TEXTURE_IMAGE_UNITS': 16,
            'GL_MAX_TEXTURE_LOD_BIAS': 15,
            'GL_MAX_TEXTURE_SIZE': 16384,
            'GL_MAX_UNIFORM_BUFFER_BINDINGS': 70,
            'GL_MAX_UNIFORM_BLOCK_SIZE': 65536,
            'GL_MAX_VARYING_COMPONENTS': 0,
            'GL_MAX_VARYING_VECTORS': 31,
            'GL_MAX_VARYING_FLOATS': 0,
            'GL_MAX_VERTEX_ATTRIBS': 16,
            'GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS': 16,
            'GL_MAX_VERTEX_UNIFORM_COMPONENTS': 4096,
            'GL_MAX_VERTEX_UNIFORM_VECTORS': 1024,
            'GL_MAX_VERTEX_OUTPUT_COMPONENTS': 128,
            'GL_MAX_VERTEX_UNIFORM_BLOCKS': 14,
            'GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET': 0,
            'GL_MAX_VERTEX_ATTRIB_BINDINGS': 0,
            'GL_VIEWPORT_BOUNDS_RANGE': (-32768, 32768),
            'GL_VIEWPORT_SUBPIXEL_BITS': 0,
            'GL_MAX_VIEWPORTS': 16
        }

.. py:attribute:: Context.includes
    :type: Dict[str, str]

    Mapping used for include statements.

.. py:attribute:: Context.extra
    :type: Any

    User defined data.

Context Flags
-------------

Context flags are used to enable or disable states in the context.
These are not the same enum values as in opengl, but are rather
bit flags so we can ``or`` them together setting multiple states
in a simple way.

These values are available in the ``Context`` object and in the
``moderngl`` module when you don't have access to the context.

.. code:: python

    import moderngl

    # From moderngl
    ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    # From context
    ctx.enable_only(ctx.DEPTH_TEST | ctx.CULL_FACE)

.. py:attribute:: Context.NOTHING
    :type: int

    Represents no states. Can be used with :py:meth:`Context.enable_only` to disable all states.

.. py:attribute:: Context.BLEND
    :type: int

    Enable/disable blending

.. py:attribute:: Context.DEPTH_TEST
    :type: int

    Enable/disable depth testing

.. py:attribute:: Context.CULL_FACE
    :type: int

    Enable/disable face culling

.. py:attribute:: Context.RASTERIZER_DISCARD
    :type: int

    Enable/disable rasterization

    Context flag: Enables ``gl_PointSize`` in vertex or geometry shaders.

    When enabled we can write to ``gl_PointSize`` in the vertex shader to specify the point size
    for each individual point.

    If this value is not set in the shader the behavior is undefined. This means the points may
    or may not appear depending if the drivers enforce some default value for ``gl_PointSize``.

.. py:attribute:: Context.PROGRAM_POINT_SIZE
    :type: int

    When disabled :py:attr:`Context.point_size` is used.

Primitive Modes
---------------

.. py:attribute:: Context.POINTS
    :type: int

    Each vertex represents a point

.. py:attribute:: Context.LINES
    :type: int

    Vertices 0 and 1 are considered a line. Vertices 2 and 3 are considered a line.
    And so on. If the user specifies a non-even number of vertices, then the extra vertex is ignored.

.. py:attribute:: Context.LINE_LOOP
    :type: int

    As line strips, except that the first and last vertices are also used as a line.
    Thus, you get n lines for n input vertices. If the user only specifies 1 vertex,
    the drawing command is ignored. The line between the first and last vertices happens
    after all of the previous lines in the sequence.

.. py:attribute:: Context.LINE_STRIP
    :type: int

    The adjacent vertices are considered lines. Thus, if you pass n vertices, you will get n-1 lines.
    If the user only specifies 1 vertex, the drawing command is ignored.

.. py:attribute:: Context.TRIANGLES
    :type: int

    Vertices 0, 1, and 2 form a triangle. Vertices 3, 4, and 5 form a triangle. And so on.

.. py:attribute:: Context.TRIANGLE_STRIP
    :type: int

    Every group of 3 adjacent vertices forms a triangle. The face direction of the
    strip is determined by the winding of the first triangle. Each successive triangle
    will have its effective face order reversed, so the system compensates for that
    by testing it in the opposite way. A vertex stream of n length will generate n-2 triangles.

.. py:attribute:: Context.TRIANGLE_FAN
    :type: int

    The first vertex is always held fixed. From there on, every group of 2 adjacent
    vertices form a triangle with the first. So with a vertex stream, you get a list
    of triangles like so: (0, 1, 2) (0, 2, 3), (0, 3, 4), etc. A vertex stream of
    n length will generate n-2 triangles.

.. py:attribute:: Context.LINES_ADJACENCY
    :type: int

    These are special primitives that are expected to be used specifically with
    geomtry shaders. These primitives give the geometry shader more vertices
    to work with for each input primitive. Data needs to be duplicated in buffers.

.. py:attribute:: Context.LINE_STRIP_ADJACENCY
    :type: int

    These are special primitives that are expected to be used specifically with
    geomtry shaders. These primitives give the geometry shader more vertices
    to work with for each input primitive. Data needs to be duplicated in buffers.

.. py:attribute:: Context.TRIANGLES_ADJACENCY
    :type: int

    These are special primitives that are expected to be used specifically with
    geomtry shaders. These primitives give the geometry shader more vertices
    to work with for each input primitive. Data needs to be duplicated in buffers.

.. py:attribute:: Context.TRIANGLE_STRIP_ADJACENCY
    :type: int

    These are special primitives that are expected to be used specifically with
    geomtry shaders. These primitives give the geometry shader more vertices
    to work with for each input primitive. Data needs to be duplicated in buffers.

.. py:attribute:: Context.PATCHES
    :type: int

    primitive type can only be used when Tessellation is active. It is a primitive
    with a user-defined number of vertices, which is then tessellated based on the
    control and evaluation shaders into regular points, lines, or triangles, depending
    on the TES's settings.


Texture Filters
~~~~~~~~~~~~~~~

Also available in the :py:class:`Context` instance
including mode details.


.. py:attribute:: Context.NEAREST
    :type: int

    Returns the value of the texture element that is nearest
    (in Manhattan distance) to the specified texture coordinates.

.. py:attribute:: Context.LINEAR
    :type: int

    Returns the weighted average of the four texture elements
    that are closest to the specified texture coordinates.
    These can include items wrapped or repeated from other parts
    of a texture, depending on the values of texture repeat mode,
    and on the exact mapping.

.. py:attribute:: Context.NEAREST_MIPMAP_NEAREST
    :type: int

    Chooses the mipmap that most closely matches the size of the
    pixel being textured and uses the ``NEAREST`` criterion (the texture
    element closest to the specified texture coordinates) to produce
    a texture value.

.. py:attribute:: Context.LINEAR_MIPMAP_NEAREST
    :type: int

    Chooses the mipmap that most closely matches the size of the pixel
    being textured and uses the ``LINEAR`` criterion (a weighted average
    of the four texture elements that are closest to the specified
    texture coordinates) to produce a texture value.

.. py:attribute:: Context.NEAREST_MIPMAP_LINEAR
    :type: int

    Chooses the two mipmaps that most closely match the size of the
    pixel being textured and uses the ``NEAREST`` criterion (the texture
    element closest to the specified texture coordinates ) to produce
    a texture value from each mipmap. The final texture value is a
    weighted average of those two values.

.. py:attribute:: Context.LINEAR_MIPMAP_LINEAR
    :type: int

    Chooses the two mipmaps that most closely match the size of the pixel
    being textured and uses the ``LINEAR`` criterion (a weighted average
    of the texture elements that are closest to the specified texture
    coordinates) to produce a texture value from each mipmap.
    The final texture value is a weighted average of those two values.


Blend Functions
---------------

Blend functions are used with :py:attr:`Context.blend_func`
to control blending operations.

.. code::

    # Default value
    ctx.blend_func = ctx.SRC_ALPHA, ctx.ONE_MINUS_SRC_ALPHA

.. py:attribute:: Context.ZERO
    :type: int

    (0,0,0,0)

.. py:attribute:: Context.ONE
    :type: int

    (1,1,1,1)

.. py:attribute:: Context.SRC_COLOR
    :type: int

    (Rs0/kR,Gs0/kG,Bs0/kB,As0/kA)

.. py:attribute:: Context.ONE_MINUS_SRC_COLOR
    :type: int

    (1,1,1,1) - (Rs0/kR,Gs0/kG,Bs0/kB,As0/kA)

.. py:attribute:: Context.SRC_ALPHA
    :type: int

    (As0/kA,As0/kA,As0/kA,As0/kA)

.. py:attribute:: Context.ONE_MINUS_SRC_ALPHA
    :type: int

    (1,1,1,1) - (As0/kA,As0/kA,As0/kA,As0/kA)

.. py:attribute:: Context.DST_ALPHA
    :type: int

    (Ad/kA,Ad/kA,Ad/kA,Ad/kA)

.. py:attribute:: Context.ONE_MINUS_DST_ALPHA
    :type: int

    (1,1,1,1) - (Ad/kA,Ad/kA,Ad/kA,Ad/kA)

.. py:attribute:: Context.DST_COLOR
    :type: int

    (Rd/kR,Gd/kG,Bd/kB,Ad/kA)

.. py:attribute:: Context.ONE_MINUS_DST_COLOR
    :type: int

    (1,1,1,1) - (Rd/kR,Gd/kG,Bd/kB,Ad/kA)


Blend Function Shortcuts
------------------------

.. py:attribute:: Context.DEFAULT_BLENDING
    :type: tuple

    Shotcut for the default blending ``SRC_ALPHA, ONE_MINUS_SRC_ALPHA``

.. py:attribute:: Context.ADDITIVE_BLENDING
    :type: tuple

    Shotcut for additive blending ``ONE, ONE``

.. py:attribute:: Context.PREMULTIPLIED_ALPHA
    :type: tuple

    Shotcut for blend mode when using premultiplied alpha ``SRC_ALPHA, ONE``


Blend Equations
---------------

Used with :py:attr:`Context.blend_equation`.

.. py:attribute:: Context.FUNC_ADD
    :type: int

    source + destination

.. py:attribute:: Context.FUNC_SUBTRACT
    :type: int

    source - destination

.. py:attribute:: Context.FUNC_REVERSE_SUBTRACT
    :type: int

    destination - source

.. py:attribute:: Context.MIN
    :type: int

    Minimum of source and destination

.. py:attribute:: Context.MAX
    :type: int

    Maximum of source and destination


Other Enums
-----------

.. py:attribute:: Context.FIRST_VERTEX_CONVENTION
    :type: int

    Specifies the first vertex should be used as the source of data for flat shaded varyings.
    Used with :py:attr:`Context.provoking_vertex`.

.. py:attribute:: Context.LAST_VERTEX_CONVENTION
    :type: int

    Specifies the last vertex should be used as the source of data for flat shaded varyings.
    Used with :py:attr:`Context.provoking_vertex`.

.. py:attribute:: Context.VERTEX_ATTRIB_ARRAY_BARRIER_BIT
    :type: int

    VERTEX_ATTRIB_ARRAY_BARRIER_BIT

.. py:attribute:: Context.ELEMENT_ARRAY_BARRIER_BIT
    :type: int

    ELEMENT_ARRAY_BARRIER_BIT

.. py:attribute:: Context.UNIFORM_BARRIER_BIT
    :type: int

    UNIFORM_BARRIER_BIT

.. py:attribute:: Context.TEXTURE_FETCH_BARRIER_BIT
    :type: int

    TEXTURE_FETCH_BARRIER_BIT

.. py:attribute:: Context.SHADER_IMAGE_ACCESS_BARRIER_BIT
    :type: int

    SHADER_IMAGE_ACCESS_BARRIER_BIT

.. py:attribute:: Context.COMMAND_BARRIER_BIT
    :type: int

    COMMAND_BARRIER_BIT

.. py:attribute:: Context.PIXEL_BUFFER_BARRIER_BIT
    :type: int

    PIXEL_BUFFER_BARRIER_BIT

.. py:attribute:: Context.TEXTURE_UPDATE_BARRIER_BIT
    :type: int

    TEXTURE_UPDATE_BARRIER_BIT

.. py:attribute:: Context.BUFFER_UPDATE_BARRIER_BIT
    :type: int

    BUFFER_UPDATE_BARRIER_BIT

.. py:attribute:: Context.FRAMEBUFFER_BARRIER_BIT
    :type: int

    FRAMEBUFFER_BARRIER_BIT

.. py:attribute:: Context.TRANSFORM_FEEDBACK_BARRIER_BIT
    :type: int

    TRANSFORM_FEEDBACK_BARRIER_BIT

.. py:attribute:: Context.ATOMIC_COUNTER_BARRIER_BIT
    :type: int

    ATOMIC_COUNTER_BARRIER_BIT

.. py:attribute:: Context.SHADER_STORAGE_BARRIER_BIT
    :type: int

    SHADER_STORAGE_BARRIER_BIT

.. py:attribute:: Context.ALL_BARRIER_BITS
    :type: int

    ALL_BARRIER_BITS

Examples
--------

ModernGL Context
~~~~~~~~~~~~~~~~

.. code-block:: python

    import moderngl
    # create a window
    ctx = moderngl.create_context()
    print(ctx.version_code)

Standalone ModernGL Context
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import moderngl
    ctx = moderngl.create_standalone_context()
    print(ctx.version_code)


--------------------------------------------------------------------------------


.. File: reference/uniform.rst

Uniform
=======

.. py:class:: Uniform

    Available in :py:meth:`Program.__getitem__`

    A uniform is a global GLSL variable declared with the 'uniform' storage qualifier.

    These act as parameters that the user of a shader program can pass to that program.

    In ModernGL, Uniforms can be accessed using :py:meth:`Program.__getitem__`
    or :py:meth:`Program.__iter__`

Methods
-------

.. py:method:: read() -> bytes:

    Read the value of the uniform.

.. py:method:: write(data: Any) -> None:

    Write the value of the uniform.

Attributes
----------

.. py:attribute:: Uniform.location
    :type: int

    The location of the uniform.
    The result of the glGetUniformLocation.

.. py:attribute:: Uniform.array_length
    :type: int

    If the uniform is an array the array_length is the length of the array otherwise `1`.

.. py:attribute:: Uniform.dimension
    :type: int

    The uniform dimension.

.. py:attribute:: Uniform.name
    :type: str

    The uniform name.

    The name does not contain leading `[0]`.
    The name may contain `[ ]` when the uniform is part of a struct.

.. py:attribute:: Uniform.value
    :type: Any

    The uniform value stored in the program object.

.. py:attribute:: Uniform.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: reference/varying.rst

Varying
=======

.. py:class:: Varying

    Available in :py:meth:`Program.__getitem__`

    Represents a program output varying.

.. py:attribute:: Varying.name
    :type: str

    The name of the varying.

.. py:attribute:: Varying.number
    :type: int

    The output location of the varying.

.. py:attribute:: Varying.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: reference/uniform_block.rst

UniformBlock
============

.. py:class:: UniformBlock

    Available in :py:meth:`Program.__getitem__`

.. py:attribute:: UniformBlock.binding
    :type: int

    The binding of the uniform block. Same as the value.

.. py:attribute:: UniformBlock.value
    :type: int

    The value of the uniform block. Same as the binding.

.. py:attribute:: UniformBlock.name
    :type: str

    The name of the uniform block.

.. py:attribute:: UniformBlock.index
    :type: int

    The index of the uniform block.

.. py:attribute:: UniformBlock.size
    :type: int

    The size of the uniform block.

.. py:attribute:: UniformBlock.extra
    :type: Any

    User defined data.


--------------------------------------------------------------------------------


.. File: topics/gc.rst

.. py:currentmodule:: moderngl


The Lifecycle of a ModernGL Object
==================================

From moderngl 5.7 we support three different garbage collection modes.
This should be configured using the :py:attr:`Context.gc_mode` attribute
preferably right after the context is created.

The current supported modes are:

* ``None``: (default) No garbage collection is performed. Objects needs to
  to be manually released like in previous versions of moderngl.
* ``"context_gc"``: Dead objects are collected in :py:attr:`Context.objects`.
  These can periodically be released using :py:meth:`Context.gc`.
* ``"auto"``: Dead objects are destroyed automatically like we would
  expect in python.

It's important to realize here that garbage collection is not about
the python objects itself, but the underlying OpenGL objects. ModernGL
operates in many different environments were garbage collection can be
a challenge. This depends on factors like who is controlling the existence
of the OpenGL context and challenges around threading in python.

Standalone / Headless Context
-----------------------------

In this instance we control when the context is created and destroyed.
Using ``"auto"`` garbage collection is perfectly reasonable in this
situation.

Context Detection
-----------------

When detecting an existing context from some window library we have no
direct control over the existence of the context. Using ``"auto"`` mode
is dangerous can can cause crashes especially on application exit.
The window and context is destroyed and closed, then moderngl will
try to destroy resources in a context that no longer exists.
Use ``"context_gc"`` mode to avoid this.

It can be possible to switch the ``gc_mode`` to ``None`` when
the window is closed. This can still be a problem if you have
race conditions due to resources being created in the render loop.

The Threading Issue
-------------------

When using threads in python the garbage collector can run in any thread.
This is a problem for OpenGL because only the main thread is allowed
to interact with the context. When using threads in your application
you should be using ``"context_gc"`` mode and periodically call ``Context.gc``
for example during every frame swap.

Manually Releasing Objects
--------------------------

Objects in moderngl don't automatically release the OpenGL resources when
``gc_mode=None`` is used.
Each type has a ``release()`` method that needs to be called to properly clean
up everything::

    # Create a texture
    texture = ctx.texture((10, 10), 4)

    # Properly release the opengl resources
    texture.release()

Detecting Released Objects
--------------------------

If you for some reason need to detect if a resource was released it can be done
by checking the type of the internal moderngl object (``.mglo`` property)::

    >> import moderngl
    >> ctx = moderngl.create_standalone_context()
    >> buffer = ctx.buffer(reserve=1024)
    >> type(buffer.mglo)
    <class 'mgl.Buffer'>
    >> buffer.release()
    >> type(buffer.mglo)
    <class '_moderngl.InvalidObject'>
    >> type(buffer.mglo) == moderngl.mgl.InvalidObject
    True


--------------------------------------------------------------------------------


.. File: topics/texture_formats.rst

.. _texture-format-label:

Texture Format
==============

.. py:currentmodule:: moderngl

Description
-----------

The format of a texture can be described by the ``dtype`` parameter
during texture creation. For example the :py:meth:`moderngl.Context.texture`.
The default ``dtype`` is ``f1``. Each component is an unsigned byte (0-255)
that is normalized when read in a shader into a value from 0.0 to 1.0.

The formats are based on the string formats used in numpy.

Some quick example of texture creation::

    # RGBA (4 component) f1 texture
    texture = ctx.texture((100, 100), 4)  # dtype f1 is default

    # R (1 component) f4 texture (32 bit float)
    texture = ctx.texture((100, 100), 1, dtype="f4")

    # RG (2 component) u2 texture (16 bit unsigned integer)
    texture = ctx.texture((100, 100), 2, dtype="u2")


Texture contents can be passed in using the ``data`` parameter during
creation or by using the ``write()`` method. The object passed in
``data`` can be bytes or any object supporting the buffer protocol.

When writing data to texture the data type can be derived from
the internal format in the tables below. ``f1`` textures takes
unsigned bytes (``u1`` or ``numpy.uint8`` in numpy) while
``f2`` textures takes 16 bit floats (``f2`` or ``numpy.float16`` in numpy).


Float Textures
--------------

``f1`` textures are just unsigned bytes (8 bits per component) (``GL_UNSIGNED_BYTE``)

The ``f1`` texture is the most commonly used textures in OpenGL
and is currently the default. Each component takes 1 byte (4 bytes for RGBA).
This is not really a "real" float format, but a shader will read
normalized values from these textures. ``0-255`` (byte range) is read
as a value from ``0.0`` to ``1.0`` in shaders.

In shaders the sampler type should be ``sampler2D``, ``sampler2DArray``
``sampler3D``, ``samplerCube`` etc.

+----------+---------------+---------------+-------------------+
| **dtype**|  *Components* | *Base Format* | *Internal Format* |
+==========+===============+===============+===================+
| f1       |  1            | GL_RED        | GL_R8             |
+----------+---------------+---------------+-------------------+
| f1       |  2            | GL_RG         | GL_RG8            |
+----------+---------------+---------------+-------------------+
| f1       |  3            | GL_RGB        | GL_RGB8           |
+----------+---------------+---------------+-------------------+
| f1       |  4            | GL_RGBA       | GL_RGBA8          |
+----------+---------------+---------------+-------------------+

``f2`` textures stores 16 bit float values (``GL_HALF_FLOAT``).

+----------+---------------+---------------+-------------------+
| **dtype**|  *Components* | *Base Format* | *Internal Format* |
+==========+===============+===============+===================+
| f2       |  1            | GL_RED        | GL_R16F           |
+----------+---------------+---------------+-------------------+
| f2       |  2            | GL_RG         | GL_RG16F          |
+----------+---------------+---------------+-------------------+
| f2       |  3            | GL_RGB        | GL_RGB16F         |
+----------+---------------+---------------+-------------------+
| f2       |  4            | GL_RGBA       | GL_RGBA16F        |
+----------+---------------+---------------+-------------------+

``f4`` textures store 32 bit float values. (``GL_FLOAT``)
Note that some drivers do not like 3 components because of alignment.

+----------+---------------+---------------+-------------------+
| **dtype**|  *Components* | *Base Format* | *Internal Format* |
+==========+===============+===============+===================+
| f4       |  1            | GL_RED        | GL_R32F           |
+----------+---------------+---------------+-------------------+
| f4       |  2            | GL_RG         | GL_RG32F          |
+----------+---------------+---------------+-------------------+
| f4       |  3            | GL_RGB        | GL_RGB32F         |
+----------+---------------+---------------+-------------------+
| f4       |  4            | GL_RGBA       | GL_RGBA32F        |
+----------+---------------+---------------+-------------------+

Integer Textures
----------------

Integer textures come in a signed and unsigned version. The advantage
with integer textures is that shader can read the raw integer values
from them using for example ``usampler*`` (unsigned) or ``isampler*``
(signed).

Integer textures do not support ``LINEAR`` filtering (only ``NEAREST``).

Unsigned
~~~~~~~~

``u1`` textures store unsigned byte values (``GL_UNSIGNED_BYTE``).

In shaders the sampler type should be ``usampler2D``, ``usampler2DArray``
``usampler3D``, ``usamplerCube`` etc.

+----------+---------------+-----------------+-------------------+
| **dtype**|  *Components* | *Base Format*   | *Internal Format* |
+==========+===============+=================+===================+
| u1       |  1            | GL_RED_INTEGER  | GL_R8UI           |
+----------+---------------+-----------------+-------------------+
| u1       |  2            | GL_RG_INTEGER   | GL_RG8UI          |
+----------+---------------+-----------------+-------------------+
| u1       |  3            | GL_RGB_INTEGER  | GL_RGB8UI         |
+----------+---------------+-----------------+-------------------+
| u1       |  4            | GL_RGBA_INTEGER | GL_RGBA8UI        |
+----------+---------------+-----------------+-------------------+

``u2`` textures store 16 bit unsigned integers (``GL_UNSIGNED_SHORT``).

+----------+---------------+-----------------+-------------------+
| **dtype**|  *Components* | *Base Format*   | *Internal Format* |
+==========+===============+=================+===================+
| u2       |  1            | GL_RED_INTEGER  | GL_R16UI          |
+----------+---------------+-----------------+-------------------+
| u2       |  2            | GL_RG_INTEGER   | GL_RG16UI         |
+----------+---------------+-----------------+-------------------+
| u2       |  3            | GL_RGB_INTEGER  | GL_RGB16UI        |
+----------+---------------+-----------------+-------------------+
| u2       |  4            | GL_RGBA_INTEGER | GL_RGBA16UI       |
+----------+---------------+-----------------+-------------------+

``u4`` textures store 32 bit unsigned integers (``GL_UNSIGNED_INT``)

+----------+---------------+-----------------+-------------------+
| **dtype**|  *Components* | *Base Format*   | *Internal Format* |
+==========+===============+=================+===================+
| u4       |  1            | GL_RED_INTEGER  | GL_R32UI          |
+----------+---------------+-----------------+-------------------+
| u4       |  2            | GL_RG_INTEGER   | GL_RG32UI         |
+----------+---------------+-----------------+-------------------+
| u4       |  3            | GL_RGB_INTEGER  | GL_RGB32UI        |
+----------+---------------+-----------------+-------------------+
| u4       |  4            | GL_RGBA_INTEGER | GL_RGBA32UI       |
+----------+---------------+-----------------+-------------------+

Signed
~~~~~~

``i1`` textures store signed byte values (``GL_BYTE``).

In shaders the sampler type should be ``isampler2D``, ``isampler2DArray``
``isampler3D``, ``isamplerCube`` etc.

+----------+---------------+-----------------+-------------------+
| **dtype**|  *Components* | *Base Format*   | *Internal Format* |
+==========+===============+=================+===================+
| i1       |  1            | GL_RED_INTEGER  | GL_R8I            |
+----------+---------------+-----------------+-------------------+
| i1       |  2            | GL_RG_INTEGER   | GL_RG8I           |
+----------+---------------+-----------------+-------------------+
| i1       |  3            | GL_RGB_INTEGER  | GL_RGB8I          |
+----------+---------------+-----------------+-------------------+
| i1       |  4            | GL_RGBA_INTEGER | GL_RGBA8I         |
+----------+---------------+-----------------+-------------------+

``i2`` textures store 16 bit integers (``GL_SHORT``).

+----------+---------------+-----------------+-------------------+
| **dtype**|  *Components* | *Base Format*   | *Internal Format* |
+==========+===============+=================+===================+
| i2       |  1            | GL_RED_INTEGER  | GL_R16I           |
+----------+---------------+-----------------+-------------------+
| i2       |  2            | GL_RG_INTEGER   | GL_RG16I          |
+----------+---------------+-----------------+-------------------+
| i2       |  3            | GL_RGB_INTEGER  | GL_RGB16I         |
+----------+---------------+-----------------+-------------------+
| i2       |  4            | GL_RGBA_INTEGER | GL_RGBA16I        |
+----------+---------------+-----------------+-------------------+

``i4`` textures store 32 bit integers (``GL_INT``)

+----------+---------------+-----------------+-------------------+
| **dtype**|  *Components* | *Base Format*   | *Internal Format* |
+==========+===============+=================+===================+
| i4       |  1            | GL_RED_INTEGER  | GL_R32I           |
+----------+---------------+-----------------+-------------------+
| i4       |  2            | GL_RG_INTEGER   | GL_RG32I          |
+----------+---------------+-----------------+-------------------+
| i4       |  3            | GL_RGB_INTEGER  | GL_RGB32I         |
+----------+---------------+-----------------+-------------------+
| i4       |  4            | GL_RGBA_INTEGER | GL_RGBA32I        |
+----------+---------------+-----------------+-------------------+

Normalized Integer Textures
---------------------------

Normalized integers are integer texture, but texel reads in a shader
returns normalized values (``[0.0, 1.0]``). For example an unsigned 16
bit fragment with the value ``2**16-1`` will be read as ``1.0``.

Normalized integer textures should use the `sampler2D` sampler
type. Also note that there's no standard for normalized 32 bit
integer textures because a float32 doesn't have enough precision
to express a 32 bit integer as a number between 0.0 and 1.0.

Unsigned
~~~~~~~~

``nu1`` textures is really the same as an ``f1``. Each component
is a ``GL_UNSIGNED_BYTE``, but are read by the shader in normalized
form ``[0.0, 1.0]``.

+----------+---------------+-----------------+-------------------+
| **dtype**|  *Components* | *Base Format*   | *Internal Format* |
+==========+===============+=================+===================+
| nu1      |  1            | GL_RED          | GL_R8             |
+----------+---------------+-----------------+-------------------+
| nu1      |  2            | GL_RG           | GL_RG8            |
+----------+---------------+-----------------+-------------------+
| nu1      |  3            | GL_RGB          | GL_RGB8           |
+----------+---------------+-----------------+-------------------+
| nu1      |  4            | GL_RGBA         | GL_RGBA8          |
+----------+---------------+-----------------+-------------------+

``nu2`` textures store 16 bit unsigned integers (``GL_UNSIGNED_SHORT``).
The value range ``[0, 2**16-1]`` will be normalized into ``[0.0, 1.0]``.

+----------+---------------+-----------------+-------------------+
| **dtype**|  *Components* | *Base Format*   | *Internal Format* |
+==========+===============+=================+===================+
| nu2      |  1            | GL_RED          | GL_R16            |
+----------+---------------+-----------------+-------------------+
| nu2      |  2            | GL_RG           | GL_RG16           |
+----------+---------------+-----------------+-------------------+
| nu2      |  3            | GL_RGB          | GL_RGB16          |
+----------+---------------+-----------------+-------------------+
| nu2      |  4            | GL_RGBA         | GL_RGBA16         |
+----------+---------------+-----------------+-------------------+

Signed
~~~~~~

``ni1`` textures store 8 bit signed integers (``GL_BYTE``).
The value range ``[0, 127]`` will be normalized into ``[0.0, 1.0]``.
Negative values will be clamped.

+----------+---------------+-----------------+-------------------+
| **dtype**|  *Components* | *Base Format*   | *Internal Format* |
+==========+===============+=================+===================+
| ni1      |  1            | GL_RED          | GL_R8             |
+----------+---------------+-----------------+-------------------+
| ni1      |  2            | GL_RG           | GL_RG8            |
+----------+---------------+-----------------+-------------------+
| ni1      |  3            | GL_RGB          | GL_RGB8           |
+----------+---------------+-----------------+-------------------+
| ni1      |  4            | GL_RGBA         | GL_RGBA8          |
+----------+---------------+-----------------+-------------------+

``ni2`` textures store 16 bit signed integers (``GL_SHORT``).
The value range ``[0, 2**15-1]`` will be normalized into ``[0.0, 1.0]``.
Negative values will be clamped.

+----------+---------------+-----------------+-------------------+
| **dtype**|  *Components* | *Base Format*   | *Internal Format* |
+==========+===============+=================+===================+
| ni2      |  1            | GL_RED          | GL_R16            |
+----------+---------------+-----------------+-------------------+
| ni2      |  2            | GL_RG           | GL_RG16           |
+----------+---------------+-----------------+-------------------+
| ni2      |  3            | GL_RGB          | GL_RGB16          |
+----------+---------------+-----------------+-------------------+
| ni2      |  4            | GL_RGBA         | GL_RGBA16         |
+----------+---------------+-----------------+-------------------+

Overriding internalformat
-------------------------

:py:meth:`Context.texture` supports overriding the internalformat
of the texture. This is only necessary when needing a different
internal formats from the tables above. This can for
example be ``GL_SRGB8 = 0x8C41`` or some compressed format.
You may also need to look up in :py:attr:`Context.extensions`
to ensure the context supports internalformat you are using.
We do not provide the enum values for these alternative internalformats.
They can be looked up in the registry : https://raw.githubusercontent.com/KhronosGroup/OpenGL-Registry/master/xml/gl.xml

Example::

    texture = ctx.texture(image.size, 3, data=srbg_data, internal_format=GL_SRGB8)


--------------------------------------------------------------------------------


.. File: topics/buffer_format.rst

.. _buffer-format-label:

Buffer Format
=============

.. py:currentmodule:: moderngl

Description
-----------

A buffer format is a short string describing the layout of data in a vertex
buffer object (VBO).

A VBO often contains a homogeneous array of C-like structures. The buffer
format describes what each element of the array looks like. For example,
a buffer containing an array of high-precision 2D vertex positions might have
the format ``"2f8"`` - each element of the array consists of two floats, each
float being 8 bytes wide, ie. a double.

Buffer formats are used in the :py:meth:`Context.vertex_array()` constructor,
as the 2nd component of the `content` arg.
See the :ref:`example-of-simple-usage-label` below.

Syntax
------

A buffer format looks like:

    ``[count]type[size] [[count]type[size]...] [/usage]``

Where:

- ``count`` is an optional integer. If omitted, it defaults to ``1``.
- ``type`` is a single character indicating the data type:

   - ``f`` float
   - ``i`` int
   - ``u`` unsigned int
   - ``x`` padding
- ``size`` is an optional number of bytes used to store the type.
  If omitted, it defaults to 4 for numeric types, or to 1 for padding bytes.

  A format may contain multiple, space-separated ``[count]type[size]`` triples
  (See the :ref:`example-of-single-interleaved-array-label`), followed by:


- ``/usage`` is optional. It should be preceded by a space, and then consists
  of a slash followed by a single character, indicating how successive values
  in the buffer should be passed to the shader:

   - ``/v`` per vertex.
     Successive values from the buffer are passed to each vertex.
     This is the default behavior if usage is omitted.
   - ``/i`` per instance.
     Successive values from the buffer are passed to each instance.
   - ``/r`` per render.
     the first buffer value is passed to every vertex of every instance.
     ie. behaves like a uniform.

  When passing multiple VBOs to a VAO, the first one must be of usage ``/v``,
  as shown in the :ref:`example-of-multiple-arrays-with-differing-usage-label`.

Valid combinations of type and size are:

+----------+---------------+-------------------+-----------------+---------+
|          |                 size                                          |
+==========+===============+===================+=================+=========+
| **type** | 1             | 2                 | 4               | 8       |
+----------+---------------+-------------------+-----------------+---------+
| f        | Unsigned byte | Half float        | Float           | Double  |
|          | (normalized)  |                   |                 |         |
+----------+---------------+-------------------+-----------------+---------+
| i        | Byte          | Short             | Int             | \-      |
+----------+---------------+-------------------+-----------------+---------+
| u        | Unsigned byte | Unsigned short    | Unsigned int    | \-      |
+----------+---------------+-------------------+-----------------+---------+
| x        | 1 byte        | 2 bytes           | 4 bytes         | 8 bytes |
+----------+---------------+-------------------+-----------------+---------+

The entry ``f1`` has two unusual properties:

1. Its type is ``f`` (for float), but it defines a buffer containing unsigned
   bytes. For this size of floats only, the values are `normalized`, ie.
   unsigned bytes from 0 to 255 in the buffer are converted to float values
   from 0.0 to 1.0 by the time they reach the vertex shader. This is intended
   for passing in colors as unsigned bytes.
2. Three unsigned bytes, with a format of ``3f1``,
   may be assigned to a ``vec3`` attribute, as one would expect.
   But, from ModernGL v6.0,
   they can alternatively be passed to a ``vec4`` attribute.
   This is intended for passing a buffer of 3-byte RGB values
   into an attribute which also contains an alpha channel.

There are no size 8 variants for types ``i`` and ``u``.

This buffer format syntax is specific to ModernGL. As seen in the usage
examples below, the formats sometimes look similar to the format strings passed
to ``struct.pack``, but that is a different syntax (documented here_.)

.. _here: https://docs.python.org/3.7/library/struct.html

Buffer formats can represent a wide range of vertex attribute formats.
For rare cases of specialized attribute formats that are not expressible
using buffer formats, there is a :py:meth:`VertexArray.bind()` method, to
manually configure the underlying OpenGL binding calls. This is not generally
recommended.

Examples
--------

Example buffer formats
......................

``"2f"`` has a count of ``2`` and a type of ``f`` (float). Hence it describes
two floats, passed to a vertex shader's ``vec2`` attribute. The size of the
floats is unspecified, so defaults to ``4`` bytes. The usage of the buffer is
unspecified, so defaults to ``/v`` (vertex), meaning each successive pair of
floats in the array are passed to successive vertices during the render call.

``"3i2/i"`` means three ``i`` (integers). The size of each integer is ``2``
bytes, ie. they are shorts, passed to an ``ivec3`` attribute.
The trailing ``/i`` means that consecutive values
in the buffer are passed to successive `instances` during an instanced render
call. So the same value is passed to every vertex within a particular instance.

Buffers contining interleaved values are represented by multiple space
separated count-type-size triples. Hence:

``"2f 3u x /v"`` means:

    * ``2f``: two floats, passed to a ``vec2`` attribute, followed by
    * ``3u``: three unsigned bytes, passed to a ``uvec3``, then
    * ``x``: a single byte of padding, for alignment.

The ``/v`` indicates successive elements in the buffer are passed to successive
vertices during the render. This is the default, so the ``/v`` could be
omitted.

.. _example-of-simple-usage-label:

Example of simple usage
.......................

Consider a VBO containing 2D vertex positions, forming a single triangle::

    # a 2D triangle (ie. three (x, y) vertices)
    verts = [
         0.0, 0.9,
        -0.5, 0.0,
         0.5, 0.0,
    ]

    # pack all six values into a binary array of C-like floats
    verts_buffer = struct.pack("6f", *verts)

    # put the array into a VBO
    vbo = ctx.buffer(verts_buffer)

    # use the VBO in a VAO
    vao = ctx.vertex_array(
        shader_program,
        [
            (vbo, "2f", "in_vert"), # <---- the "2f" is the buffer format
        ]
        index_buffer_object
    )

The line ``(vbo, "2f", "in_vert")``, known as the VAO content, indicates that
``vbo`` contains an array of values, each of which consists of two floats.
These values are passed to an ``in_vert`` attribute,
declared in the vertex shader as::

    in vec2 in_vert;

The ``"2f"`` format omits a ``size`` component, so the floats default to
4-bytes each. The format also omits the trailing ``/usage`` component, which
defaults to ``/v``, so successive (x, y) rows from the buffer are passed to
successive vertices during the render call.

.. _example-of-single-interleaved-array-label:

Example of single interleaved array
...................................

A buffer array might contain elements consisting of multiple interleaved
values.

For example, consider a buffer array, each element of which contains a 2D
vertex position as floats, an RGB color as unsigned ints, and a single byte of
padding for alignment:

+-------+-------+----------+----------+----------+---------+
| position      | color                          | padding |
+-------+-------+----------+----------+----------+---------+
| x     | y     | r        | g        | b        | \-      |
+-------+-------+----------+----------+----------+---------+
| float | float | unsigned | unsigned | unsigned | byte    |
|       |       | byte     | byte     | byte     |         |
+-------+-------+----------+----------+----------+---------+

Such a buffer, however you choose to construct it, would then be passed into
a VAO using::

    vao = ctx.vertex_array(
        shader_program,
        [
            (vbo, "2f 3f1 x", "in_vert", "in_color")
        ]
        index_buffer_object
    )

The format starts with ``2f``, for the two position floats, which will
be passed to the shader's ``in_vert`` attribute, declared as::

    in vec2 in_vert;

Next, after a space, is ``3f1``, for the three color unsigned bytes, which
get normalized to floats by ``f1``. These floats will be passed to the shader's
``in_color`` attribute::

    in vec3 in_color;

Finally, the format ends with ``x``, a single byte of padding, which needs
no shader attribute name.

.. _example-of-multiple-arrays-with-differing-usage-label:

Example of multiple arrays with differing ``/usage``
....................................................

To illustrate the trailing ``/usage`` portion, consider rendering a dozen cubes
with instanced rendering. We will use:

* ``vbo_verts_normals`` contains vertices (3 floats) and normals (3 floats)
  for the vertices within a single cube.
* ``vbo_offset_orientation`` contains offsets (3 floats) and orientations (9
  float matrices) that are used to position and orient each cube.
* ``vbo_colors`` contains colors (3 floats). In this example, there is only
  one color in the buffer, that will be used for every vertex of every cube.

Our shader will take all the above values as attributes.

We bind the above VBOs in a single VAO, to prepare for an instanced rendering
call::

    vao = ctx.vertex_array(
        shader_program,
        [
            (vbo_verts_normals,      "3f 3f /v", "in_vert", "in_norm"),
            (vbo_offset_orientation, "3f 9f /i", "in_offset", "in_orientation"),
            (vbo_colors,             "3f /r",    "in_color"),
        ]
        index_buffer_object
    )

So, the vertices and normals, using ``/v``, are passed to each vertex within
an instance. This fulfills the rule that the first VBO in a VAO must have usage
``/v``. These are passed to vertex attributes as::

    in vec3 in_vert;
    in vec3 in_norm;

The offsets and orientations pass the same value to each vertex within an
instance, but then pass the next value in the buffer to the vertices of the
next instance. Passed as::

    in vec3 in_offset;
    in mat3 in_orientation;

The single color is passed to every vertex of every instance.
If we had stored the color with ``/v`` or ``/i``, then we would have had to
store duplicate identical color values in vbo_colors - one per instance or
one per vertex. To render all our cubes in a single color, this is needless
duplication. Using ``/r``, only one color is require the buffer, and it is
passed to every vertex of every instance for the whole render call::

    in vec3 in_color;

An alternative approach would be to pass in the color as a uniform, since
it is constant. But doing it as an attribute is more flexible. It allows us to
reuse the same shader program, bound to a different buffer, to pass in color
data which varies per instance, or per vertex.

.. toctree::
    :maxdepth: 2


--------------------------------------------------------------------------------


.. File: topics/context.rst


.. _context:

Context Creation
================

.. py:currentmodule:: moderngl

.. Note:: From moderngl 5.6 context creation is handled by the glcontext_ package.
          This makes expanding context support easier for users lowering the
          bar for contributions. It also means context creation is no longer
          limited by a moderngl releases.

.. Note:: This page might not list all supported backends as the glcontext_
          project keeps evolving. If using anything outside of the default
          contexts provided per OS, please check the listed backends in
          the glcontext_ project.

Introduction
------------

A context is an object giving moderngl access to opengl instructions
(greatly simplified). How a context is created depends on your
operating system and what kind of platform you want to target.

In the vast majority of cases you'll be using the default context
backend supported by your operating system. This backend will be
automatically selected unless a specific ``backend`` parameter is used.

Default backend per OS

* **Windows**: wgl / opengl32.dll
* **Linux**: x11/glx/libGL
* **OS X**: CGL

These default backends support two modes:

* Detecting an exiting active context possibly created by a window
  library such as glfw, sdl2, pyglet etc.
* Creating a headless context (No visible window)

Detecting an existing active context created by a window library::

    import moderngl
    # Create the window with an OpenGL context (Most window libraries support this)
    ctx = moderngl.create_context()
    # If successful we can now render to the window
    print("Default framebuffer is:", ctx.screen)

A great reference using various window libraries can be found here: 
https://github.com/moderngl/moderngl-window/tree/master/moderngl_window/context

Creating a headless context::

    import moderngl
    # Create the context
    ctx = moderngl.create_context(standalone=True)
    # Create a framebuffer we can render to
    fbo = ctx.simple_framebuffer((100, 100), 4)
    fbo.use()

Require a minimum OpenGL version
--------------------------------

ModernGL only support 3.3+ contexts. By default version 3.3
is passed in as the minimum required version of the context
returned by the backend.

To require a specific version::

    moderngl.create_context(require=430)

This will require OpenGL 4.3. If a lower context version is
returned the context creation will fail.

This attribute can be accessed in :py:attr:`Context.version_code`
and will be updated to contain the actual version code of the
context (If higher than required).

Specifying context backend
--------------------------

A ``backend`` can be passed in for more advanced usage.

For example: Making a headless EGL context on linux::

    ctx = moderngl.create_context(standalone=True, backend='egl')

.. Note:: Each backend supports additional keyword arguments for
          more advanced configuration. This can for example be
          the exact name of the library to load. More information
          in the glcontext_ docs.

Context Sharing
---------------

.. Warning:: Object sharing is an experimental feature

Some context support the ``share`` parameters enabling
object sharing between contexts. This is not needed
if you are attaching to existing context with share mode enabled.
For example if you create two windows with glfw enabling object sharing.

ModernGL objects (such as :py:class:`moderngl.Buffer`, :py:class:`moderngl.Texture`, ..)
has a ``ctx`` property containing the context they were created in.
Still **ModernGL do not check what context is currently active when
accessing these objects.** This means the object can be used
in both contexts when sharing is enabled.

This should in theory work fine with object sharing enabled::

    data1 = numpy.array([1, 2, 3, 4], dtype='u1')
    data2 = numpy.array([4, 3, 2, 1], dtype='u1')

    ctx1 = moderngl.create_context(standalone=True)
    ctx2 = moderngl.create_context(standalone=True, share=True)

    with ctx1 as ctx:
        b1 = ctx.buffer(data1)

    with ctx2 as ctx:
        b2 = ctx.buffer(data2)

    print(b1.glo)  # Displays: 1
    print(b2.glo)  # Displays: 2

    with ctx1:
        print(b1.read())
        print(b2.read())

    with ctx2:
        print(b1.read())
        print(b2.read())

Still, there are some limitations to object sharing. Especially
objects that reference other objects (framebuffer, vertex array object, etc.)

More information for a deeper dive:

* https://www.khronos.org/opengl/wiki/OpenGL_Object#Object_Sharing
* https://www.khronos.org/opengl/wiki/Memory_Model

Context Info
------------

Various information such as limits and driver information can be found in the
:py:attr:`~moderngl.Context.info` property. It can often be useful to know
the vendor and render for the context::

    >>> import moderngl
    >>> ctx = moderngl.create_context(standalone=True, gl_version=(4.6))
    >>> ctx.info["GL_VENDOR"]
    'NVIDIA Corporation'
    >>> ctx.info["GL_RENDERER"] 
    'GeForce RTX 2080 SUPER/PCIe/SSE2'
    >>> ctx.info["GL_VERSION"]  
    '3.3.0 NVIDIA 456.71'

Note that it reports version 3.3 here because ModernGL by default
requests a version 3.3 context (minimum requirement).

.. _glcontext: https://github.com/moderngl/glcontext


--------------------------------------------------------------------------------

