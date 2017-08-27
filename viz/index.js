const resl = require('resl')
const css = require('dom-css')
const fit = require('canvas-fit')
const control = require('control-panel')

// setup canvas and camera
const canvas = document.body.appendChild(document.createElement('canvas'))
const regl = require('regl')(canvas)
const camera = require('./camera.js')(canvas, {scale: true, rotate: true})
window.addEventListener('resize', fit(canvas), false)

// import draw functions
const drawBackground = require('./draw/background')(regl)
const drawSpots = require('./draw/spots')(regl)
const drawRegions = require('./draw/regions')(regl)

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

// load assets and draw
resl({
  manifest: {
    'background': {
      type: 'image',
      src: '../example/background.png'
    },

    'spots': {
      type: 'text',
      src: '../example/spots.json',
      parser: JSON.parse
    },

    'regions': {
      type: 'text',
      src: '../example/regions.json',
      parser: JSON.parse
    }
  },

  onDone: ({background, spots, regions}) => {
    const width = background.naturalWidth
    const height = background.naturalHeight
    const scale = height/width
    const texture = regl.texture(background)

    var xy
    const positions = spots.map(function (spot) {
      xy = spot.geometry.coordinates
      return [xy[1] / (width / 2) - 1.0, xy[0] / (height / 2) - 1]
    })

    const vertices = regions.map(function (region) {
      return region.geometry.coordinates.map(function (xy) {
        return [xy[1] / (width / 2) - 1.0, xy[0] / (height / 2) - 1]
      })
    })

    regl.frame(() => {
      regl.clear({
        depth: 1,
        color: [0, 0, 0, 1]
      })
  
      if (state.show) {
        drawSpots({
          distance: camera.distance, 
          color: state.color,
          positions: positions,
          count: positions.length,
          view: camera.view(),
          scale: scale
        })
      }
      
      drawBackground({
        background: texture,
        view: camera.view(),
        scale: scale
      })

      drawRegions(vertices.map(function (v) {
        return {
          distance: camera.distance, 
          color: state.color,
          vertices: v,
          count: v.length,
          view: camera.view(),
          scale: scale
        }})
      )

      camera.tick()
    })
  }
})