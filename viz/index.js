const css = require('dom-css')
const panzoom = require('pan-zoom')
const rune = require('rune.js')

const regions = require('./example/regions.json')
const spots = require('./example/spots.json')

var scale = 0.7

var width, height
var img = new Image()
img.src = '../example/background.png'
img.addEventListener('load', function(e) {
	width = img.width * scale
	height = img.height * scale
  	setup()
}, false)

function setup() {
	var container = document.createElement('div')
	document.body.append(container)
	css(container, {
		width: width,
		height: height,
		position: 'absolute'
	})

	container.append(img)
	css(img, {
		width: width,
		height: height,
		position: 'absolute',
		'z-index': -1000
	})

	var svg = document.createElement('svg')
	container.append(svg)

	// panzoom(img, function (e) {
	// 	css(img, {'transform': 'scale(' + e.dz + ')'})
	// 	console.log('translate(' + e.x + 'px, ' + e.y + 'px) scale(' + e.dz + ')')
	// })

	console.log(spots)
	console.log(regions)

	var r = new rune({
	  container: container,
	  width: width,
	  height: height
	});

	var group = r.group(0, 0);

	var xy, rad
	spots.forEach(function (spot) {
		xy = spot.geometry.coordinates
		rad = spot.properties.radius
		r.circle(xy[1] * scale, xy[0] * scale, rad, group)
			.fill(false)
			.stroke(150,150,150)
	})

	var verts, start
	regions.forEach(function (region) {
		verts = region.geometry.coordinates
		console.log(verts)
		start = r.polygon(0, 0)
			.stroke(150,150,150)
  			.fill(false)
		verts.forEach(function (v) {
			start.lineTo(v[1] * scale, v[0] * scale)
		})
	})

	r.draw()
}

// get pan and zoom working

// render points with pan and zoom

// render polygons with pan and zoom