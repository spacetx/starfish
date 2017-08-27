const mat4 = require('gl-mat4')

module.exports = function (regl) {
  return regl({
    frag: `
    precision mediump float;
    uniform sampler2D texture;
    varying vec2 uv;
    void main () {
      gl_FragColor = texture2D(texture, uv);
    }`,

    vert: `
    precision mediump float;
    uniform mat4 projection, view;
    attribute vec2 position;
    varying vec2 uv;
    void main () {
      uv = position;
      gl_Position = projection * view * vec4(-(1.0 - 2.0 * position.x), 1.0 - 2.0 * position.y, 0, 1);
    }`,

    attributes: {
      position: [
        -2, 0,
        0, -2,
        2, 2]
    },

    uniforms: {
      texture: regl.prop('background'),
      view: regl.prop('view'),
      projection: (context, props) =>
        mat4.perspective([],
          Math.PI / 2,
          context.viewportWidth * props.scale / context.viewportHeight,
          0.01, 1000),
    },

     blend: {
      enable: false,
      func: {
        srcRGB: 'one',
        srcAlpha: 'one',
        dstRGB: 'one minus src alpha',
        dstAlpha: 'one minus dst alpha',
      }
    },

    count: 3
  })
}