/**
 * SmartTalker — VRM Avatar Module
 *
 * Uses Three.js and @pixiv/three-vrm to render a 3D avatar on a canvas.
 * Implements real-time lip synchronicity via Web Audio API analysis
 * acting on the TTS response audio stream.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

// Expose to global scope so app.js can call it without being a module itself
window.VRMAvatar = {
    init,
    load,
    setEmotion,
    startLipSync
};

let scene, camera, renderer;
let currentVrm = null;
let mixer = null;
const clock = new THREE.Clock();
let animationFrameId = null;

// Lip-sync state
let audioCtx = null;
let analyser = null;
let sourceNode = null;
let isLipSyncActive = false;

/**
 * Initialize the Three.js scene
 */
function init(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`Canvas #${canvasId} not found`);
        return;
    }

    // Renderer
    renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    renderer.setClearColor(0x000000, 0); // Transparent background

    // Scene
    scene = new THREE.Scene();

    // Camera
    camera = new THREE.PerspectiveCamera(30.0, canvas.clientWidth / canvas.clientHeight, 0.1, 20.0);
    camera.position.set(0.0, 1.4, 2.5); // Focus on upper body / face

    // Lights
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
    hemiLight.position.set(0, 20, 0);
    scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(1, 2, 3);
    scene.add(dirLight);

    // Handle resize
    window.addEventListener('resize', () => {
        camera.aspect = canvas.clientWidth / canvas.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    });

    animate();
}

/**
 * Load a VRM model from a URL
 */
function load(url) {
    console.log("Loading VRM:", url);
    const loader = new GLTFLoader();

    // Install VRM plugin
    loader.register((parser) => new VRMLoaderPlugin(parser));

    loader.load(url,
        (gltf) => {
            const vrm = gltf.userData.vrm;
            if (!vrm) {
                console.error("Loaded GLTF does not contain VRM data");
                return;
            }

            // Remove previous model if exists
            if (currentVrm) {
                scene.remove(currentVrm.scene);
                VRMUtils.deepDispose(currentVrm.scene);
            }

            VRMUtils.removeUnnecessaryVertices(gltf.scene);
            VRMUtils.removeUnnecessaryJoints(gltf.scene);

            vrm.scene.rotation.y = Math.PI; // Face the camera
            scene.add(vrm.scene);

            currentVrm = vrm;

            // Optional: reset spring bones
            if (vrm.springBoneManager) {
                vrm.springBoneManager.reset();
            }

            console.log("VRM loaded successfully");
            setEmotion("neutral");
        },
        (progress) => { /* console.log(progress.loaded / progress.total * 100 + '%'); */ },
        (error) => { console.error("Error loading VRM:", error); }
    );
}

/**
 * Set avatar facial expression (emotion mapping)
 */
function setEmotion(emotionStr) {
    if (!currentVrm || !currentVrm.expressionManager) return;

    // Reset all expressions
    const presetNames = ['neutral', 'happy', 'angry', 'sad', 'relaxed', 'surprised', 'aa', 'ih', 'ou', 'ee', 'oh', 'blink', 'blinkLeft', 'blinkRight'];
    presetNames.forEach(preset => currentVrm.expressionManager.setValue(preset, 0));

    // Map common emotion names to VRM presets
    let vrmPreset = 'neutral';
    switch (emotionStr.toLowerCase()) {
        case 'happy': vrmPreset = 'happy'; break;
        case 'sad': vrmPreset = 'sad'; break;
        case 'angry': vrmPreset = 'angry'; break;
        case 'surprised': vrmPreset = 'surprised'; break;
        case 'fearful': vrmPreset = 'sad'; break;
        case 'disgusted': vrmPreset = 'angry'; break;
        default: vrmPreset = 'neutral'; break;
    }

    currentVrm.expressionManager.setValue(vrmPreset, 1.0);
}

/**
 * Setup Web Audio API to analyze TTS output and drive lip sync
 */
function startLipSync(audioElementId) {
    if (isLipSyncActive) return;

    const audioEl = document.getElementById(audioElementId);
    if (!audioEl) {
        console.warn(`Audio element #${audioElementId} not found`);
        return;
    }

    try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AudioContext();
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.5;

        // If cross-origin fails, you might need crossorigin="anonymous" on the <audio>
        sourceNode = audioCtx.createMediaElementSource(audioEl);
        sourceNode.connect(analyser);
        analyser.connect(audioCtx.destination);

        isLipSyncActive = true;
        console.log("Audio analyzer hooked for VRM lip-sync");
    } catch (err) {
        console.error("Failed to init lip-sync:", err);
    }
}

/**
 * Compute average volume from analyser and map it to jaw opening ('aa')
 */
function updateLipSync() {
    if (!isLipSyncActive || !analyser || !currentVrm || !currentVrm.expressionManager) return;

    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);

    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i];
    }
    const average = sum / dataArray.length;

    // Map volume (0-255) to blendshape weight (0-1)
    // Audio from TTS usually averages around 20-50 for speech, cap max mouth open at ~40 average
    let weight = average / 40.0;

    // Clamp between 0 and 1
    weight = Math.max(0, Math.min(1, weight));

    // Add a tiny noise curve to make it look loose and organic instead of purely robotic
    if (weight > 0.05) {
        weight = weight * (0.8 + Math.random() * 0.4);
    } else {
        weight = 0; // Close mouth completely if basically silent
    }

    // Set 'aa' expression for generic talking
    currentVrm.expressionManager.setValue('aa', Math.min(1, weight));
}

/**
 * Main render loop
 */
function animate() {
    animationFrameId = requestAnimationFrame(animate);

    const deltaTime = clock.getDelta();

    // Subtle idle animation (breathing)
    if (currentVrm) {
        const s = Math.sin(Math.PI * clock.elapsedTime);
        const headNode = currentVrm.humanoid.getNormalizedBoneNode('head');
        const chestNode = currentVrm.humanoid.getNormalizedBoneNode('chest');

        if (chestNode) {
            chestNode.rotation.x = s * 0.02; // Breathe
            chestNode.rotation.y = Math.cos(clock.elapsedTime * 0.5) * 0.01;
        }
        if (headNode) {
            headNode.rotation.y = Math.sin(clock.elapsedTime * 0.2) * 0.05;
            headNode.rotation.z = Math.cos(clock.elapsedTime * 0.3) * 0.02;
        }

        // Random blinking
        if (Math.random() < 0.01) {
            currentVrm.expressionManager.setValue('blink', 1.0);
            setTimeout(() => {
                if (currentVrm && currentVrm.expressionManager) {
                    currentVrm.expressionManager.setValue('blink', 0.0);
                }
            }, 100);
        }

        // Apply audio lip sync
        updateLipSync();

        currentVrm.update(deltaTime);
    }

    if (mixer) {
        mixer.update(deltaTime);
    }

    renderer.render(scene, camera);
}
