import Foundation
import AVFoundation
import SwiftUI


let USE_FRONT = false
let CAMERAS = (front: (cam: AVFoundation.AVCaptureDevice.DeviceType.builtInWideAngleCamera,
                       pos: AVFoundation.AVCaptureDevice.Position.front),
               back: (cam: AVFoundation.AVCaptureDevice.DeviceType.builtInDualCamera,
                      pos: AVFoundation.AVCaptureDevice.Position.back))
    


struct StreamView: UIViewControllerRepresentable {
    let cubeFaces: CubeFaces
    
    func makeUIViewController(context: Context) -> UIViewController {
        return StreamViewController(cfVM: cubeFaces)
    }
    
    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
    }
}


class CubeFaces: ObservableObject {
    @Published var faces: [Int32] = Array(repeating: 5, count: 54) {
        didSet {
            print(self.faces)
        }
    }
    
    
}


class StreamViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    let cubeFaces: CubeFaces!
    
    init(cfVM: CubeFaces) {
        self.cubeFaces = cfVM
        
        super.init(nibName: nil, bundle: nil)
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented") // or see Roman Sausarnes's answer
    }
    
    var permissionGranted = false
    let captureSession = AVCaptureSession()
    let sessionQueue = DispatchQueue(label: "sessionQueue")
    
    private var previewLayer: AVCaptureVideoPreviewLayer!
    
    var screenRect: CGRect! = nil
    
    private let videoDataOutput = AVCaptureVideoDataOutput()
    
    func checkPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            self.permissionGranted = true
            
        case .notDetermined:
            self.requestPermission()
            
        default:
            self.permissionGranted = false
        }
    }
    
    func requestPermission() {
        sessionQueue.suspend()
        AVCaptureDevice.requestAccess(for: .video) { [unowned self] granted in
            self.permissionGranted = granted
            self.sessionQueue.resume()
        }
    }
    func setupCaptureSession() {
        guard let videoDevice = AVCaptureDevice.default(USE_FRONT ? CAMERAS.front.cam : CAMERAS.back.cam,
                                                        for: .video,
                                                        position: USE_FRONT ? CAMERAS.front.pos : CAMERAS.back.pos) else { return }
        guard let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice) else { return }
        
        guard self.captureSession.canAddInput(videoDeviceInput) else { return }
        self.captureSession.addInput(videoDeviceInput)
    }
    
    override func viewDidLoad() {
        checkPermission()
        
        sessionQueue.async { [unowned self] in
            guard permissionGranted else { return }
            self.setupCaptureSession()
            self.captureSession.startRunning()
        }
        
        screenRect = UIScreen.main.bounds
        
        previewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
        previewLayer.frame = CGRect(x: 0, y: 0, width: screenRect.size.width, height: screenRect.size.height)
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        
        previewLayer.connection?.videoOrientation = .portrait
        
        getFrames()
    }
    
    private func getFrames() {
        videoDataOutput.videoSettings = [(kCVPixelBufferPixelFormatTypeKey as NSString) : NSNumber(value: kCVPixelFormatType_32BGRA)] as [String : Any]
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.frame.processing.queue"))
        self.captureSession.addOutput(videoDataOutput)
        guard let connection = self.videoDataOutput.connection(with: AVMediaType.video),
          connection.isVideoOrientationSupported else { return }
        connection.videoOrientation = .portrait
    }
    
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection) {
            print("did receive frame")
            
            guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            CVPixelBufferLockBaseAddress(imageBuffer, CVPixelBufferLockFlags.readOnly)
            let baseAddress = CVPixelBufferGetBaseAddress(imageBuffer)
            let bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer)
            let width = CVPixelBufferGetWidth(imageBuffer)
            let height = CVPixelBufferGetHeight(imageBuffer)
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            var bitmapInfo: UInt32 = CGBitmapInfo.byteOrder32Little.rawValue
            bitmapInfo |= CGImageAlphaInfo.premultipliedFirst.rawValue & CGBitmapInfo.alphaInfoMask.rawValue
            let context = CGContext(data: baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo)
            guard let quartzImage = context?.makeImage() else { return }
            CVPixelBufferUnlockBaseAddress(imageBuffer, CVPixelBufferLockFlags.readOnly)
            let image = UIImage(cgImage: quartzImage)
            
            let ret: UnsafeMutablePointer<Int32>? = Bridge().cubecv(image)
            
            DispatchQueue.main.async { [weak self, cubeFaces] in
                cubeFaces?.faces = Array(UnsafeBufferPointer(start: ret, count: 54))
                
                self!.view.layer.addSublayer(self!.previewLayer)
            }
        }
}
