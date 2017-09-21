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
const drawOutlines = require('./draw/outlines')(regl)

// setup control panel and state
var state = {
  color: [0.6, 0.2, 0.9],
  showSpots: true,
  showRegions: true
}
var panel = control([
  {type: 'color', label: 'dot color', format: 'array', initial: state.color},
  {type: 'checkbox', label: 'show spots', initial: state.showSpots},
  {type: 'checkbox', label: 'show regions', initial: state.showRegions}
],
  {theme: 'dark', position: 'top-left'}
)
panel.on('input', function (data) {
  state.color = typeof(data['dot color']) == 'string' ? state.color : data['dot color']
  state.showSpots = data['show spots']
  state.showRegions = data['show regions']
})

// load assets and draw
resl({
  manifest: {
    'background': {
      type: 'image',
      src: '../example_2/background.png'
    },

    'spots': {
      type: 'text',
      src: '../example_2/spots.json',
      parser: JSON.parse
    },

    'regions': {
      type: 'text',
      src: '../example_2/regions.json',
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
  
      if (state.showSpots) {
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

      if (state.showRegions) {
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

        drawOutlines(vertices.map(function (v) {
          return {
            distance: camera.distance, 
            color: state.color,
            vertices: v,
            count: v.length,
            view: camera.view(),
            scale: scale
          }})
        )
      }

      camera.tick()
    })
  }
})