const resl = require('resl')
const css = require('dom-css')
const mat4 = require('gl-mat4')
const fit = require('canvas-fit')
const control = require('control-panel')

// setup canvas and camera
const canvas = document.body.appendChild(document.createElement('canvas'))
const regl = require('regl')(canvas)
const camera = require('./camera.js')(canvas, {scale: false, rotate: true})
window.addEventListener('resize', fit(canvas), false)

// load spots
var spots = require('./example/spots.json')
spots = spots.map(function (spot) {
  var xy = spot.geometry.coordinates
  return [xy[1] / 665 - 1.0, xy[0] / 490 - 1]
})

// setup control panel and state
var state = {
  show:  true,
  color: [0.6, 0.2, 0.9]
}
var panel = control([
  {type: 'color', label: 'dot color', format: 'array', initial: state.color},
  {type: 'checkbox', label: 'show dots', initial: state.show}
],
  {theme: 'dark', position: 'top-left'}
)
panel.on('input', function (data) {
  state.color = typeof(data['dot color']) == 'string' ? state.color : data['dot color']
  state.show = data['show dots']
})

// create regl spot drawing function
const drawSpots = regl({
  vert: `
  precision mediump float;
  attribute vec2 position;
  uniform float distance;
  uniform vec3 color;
  uniform mat4 projection, view;
  varying vec3 fragColor;
  void main() {
    gl_PointSize = 6.0 / pow(distance, 0.5);
    gl_Position = view * vec4(position.x, -position.y, 0, 1);
    fragColor = color;
  }`,

  frag: `
  precision lowp float;
  varying vec3 fragColor;
  void main() {
    if (length(gl_PointCoord.xy - 0.5) > 0.5) {
      discard;
    }
    gl_FragColor = vec4(fragColor, 1);
  }`,

  attributes: {
    position: spots
  },

  uniforms: {
    color: regl.prop('color'),
    distance: regl.prop('distance'),
    view: () => camera.view(),
    projection: ({viewportWidth, viewportHeight}) =>
      mat4.perspective([],
        Math.PI / 2,
        viewportWidth / viewportHeight,
        0.01,
        1000),
  },

  count: spots.length,

  primitive: 'points'
})

// create regl function for drawing background
var drawBackground = regl({
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
    gl_Position = view * vec4(-(1.0 - 2.0 * position.x), 1.0 - 2.0 * position.y, 0, 1);
  }`,

  attributes: {
    position: [
      -2, 0,
      0, -2,
      2, 2]
  },

  uniforms: {
    texture: regl.prop('background'),
    view: () => camera.view(),
    projection: ({viewportWidth, viewportHeight}) =>
      mat4.perspective([],
        Math.PI / 2,
        viewportWidth / viewportHeight,
        0.01,
        1000),
  },

  count: 3
})

// load background and draw
resl({
  manifest: {
    'background': {
      type: 'image',
      src: '../example/background.png'
    },
  },

  onDone: ({background}) => {
    const texture = regl.texture(background)

    regl.frame(() => {
      regl.clear({
        depth: 1,
        color: [0, 0, 0, 1]
    })
    if (state.show) {
      drawSpots({distance: camera.distance, color: state.color})
    }
      drawBackground({background: texture})
      camera.tick()
      console.log(camera.view())
    })
  }
})