const mat4 = require('gl-mat4')

module.exports = function (regl) {
  return regl({
    vert: `
    precision mediump float;
    attribute vec2 vertex;
    uniform float distance;
    uniform vec3 color;
    uniform mat4 projection, view;
    varying vec3 fragColor;
    void main() {
      gl_Position = projection * view * vec4(vertex.x, -vertex.y, 0, 1);
      fragColor = color;
    }`,

    frag: `
    precision lowp float;
    varying vec3 fragColor;
    void main() {
      gl_FragColor = vec4(fragColor, 0.5);
    }`,

    attributes: {
      vertex: regl.prop('vertices')
    },

    uniforms: {
      color: regl.prop('color'),
      distance: regl.prop('distance'),
      view: regl.prop('view'),
      projection: (context, props) =>
        mat4.perspective([],
          Math.PI / 2,
          context.viewportWidth * props.scale / context.viewportHeight,
          0.01, 1000),
    },

    count: regl.prop('count'),

    primitive: 'triangle fan',

    depth: {
      enable: false,
    },

    blend: {
      enable: true,
      func: {
        src: 'src alpha',
        dst: 1
      }
    },
  })
}