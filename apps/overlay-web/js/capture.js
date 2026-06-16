// Capture source adapter. Stage 1 = WebcamSource. Builds 2/3 (screen region, native
// draw-anywhere) implement the same interface: start() -> HTMLVideoElement, stop().
export class WebcamSource {
  constructor(video) { this.video = video; this.stream = null; }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, frameRate: 30 }, audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    return this.video;
  }

  stop() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }
}
