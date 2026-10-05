"use client";

import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { type RefObject, useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

import type { Mesh, Timeline } from "@/lib/api";
import { colorParcels, sampleTimeline } from "@/lib/brain";

export type ViewPreset = "left" | "right" | "top" | "back";

// fsaverage coordinates are RAS (x right, y anterior, z superior). The mesh group is rotated
// -90° about X so z points up on screen; after that, anterior points toward -Z.
const VIEWS: Record<ViewPreset, [number, number, number]> = {
  left: [-400, 30, -10],
  right: [400, 30, -10],
  top: [0, 420, 1],
  back: [0, 50, 420],
};
const MEDIAL_WALL = [0.22, 0.23, 0.27];

type Props = {
  mesh: Mesh;
  timeline: Timeline | null;
  range: [number, number];
  audioRef: RefObject<HTMLAudioElement | null>;
  view: ViewPreset;
};

function CortexMesh({ mesh, timeline, range, audioRef }: Omit<Props, "view">) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(mesh.positions, 3));
    g.setIndex(new THREE.BufferAttribute(mesh.faces, 1));
    const colors = new Float32Array(mesh.nVertices * 3);
    for (let v = 0; v < mesh.nVertices; v++) colors.set(MEDIAL_WALL, v * 3);
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage));
    g.computeVertexNormals();
    return g;
  }, [mesh]);
  useEffect(() => () => geometry.dispose(), [geometry]);

  // Scratch buffers, allocated once per mesh: no allocations inside the frame loop.
  const scratch = useMemo(
    () => ({ values: new Float32Array(mesh.nParcels), rgb: new Float32Array(mesh.nParcels * 3) }),
    [mesh],
  );
  const last = useRef<{ t: number; tl: Timeline | null; range: [number, number] | null }>({ t: -1, tl: null, range: null });

  useFrame(() => {
    const colorAttr = geometry.getAttribute("color") as THREE.BufferAttribute;
    const colors = colorAttr.array as Float32Array;
    const t = audioRef.current?.currentTime ?? 0;
    const prev = last.current;
    if (prev.tl === timeline && prev.range === range && Math.abs(prev.t - t) < 1 / 60) return;
    last.current = { t, tl: timeline, range };

    const parcel = mesh.parcel;
    if (!timeline) {
      for (let v = 0; v < mesh.nVertices; v++) colors.set(MEDIAL_WALL, v * 3);
    } else {
      sampleTimeline(timeline, t, scratch.values);
      colorParcels(scratch.values, range[0], range[1], scratch.rgb);
      const rgb = scratch.rgb;
      for (let v = 0; v < mesh.nVertices; v++) {
        const p = parcel[v];
        const o = v * 3;
        if (p < 0) {
          colors[o] = MEDIAL_WALL[0];
          colors[o + 1] = MEDIAL_WALL[1];
          colors[o + 2] = MEDIAL_WALL[2];
        } else {
          colors[o] = rgb[p * 3];
          colors[o + 1] = rgb[p * 3 + 1];
          colors[o + 2] = rgb[p * 3 + 2];
        }
      }
    }
    colorAttr.needsUpdate = true;
  });

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <mesh geometry={geometry}>
        <meshStandardMaterial vertexColors roughness={0.75} metalness={0.0} />
      </mesh>
    </group>
  );
}

function CameraRig({ view }: { view: ViewPreset }) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null;
  useEffect(() => {
    camera.position.set(...VIEWS[view]);
    camera.lookAt(0, 0, 0);
    if (controls) {
      controls.target.set(0, 0, 0);
      controls.update();
    }
  }, [view, camera, controls]);
  return null;
}

export default function BrainViewer({ mesh, timeline, range, audioRef, view }: Props) {
  return (
    <Canvas camera={{ position: VIEWS[view], fov: 38, near: 1, far: 2000 }} dpr={[1, 2]} gl={{ antialias: true }}>
      <color attach="background" args={["#181818"]} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[200, 300, 200]} intensity={1.1} />
      <directionalLight position={[-250, -100, -150]} intensity={0.45} />
      <CortexMesh mesh={mesh} timeline={timeline} range={range} audioRef={audioRef} />
      <OrbitControls makeDefault enablePan={false} minDistance={140} maxDistance={700} />
      <CameraRig view={view} />
    </Canvas>
  );
}
