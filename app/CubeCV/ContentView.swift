import SwiftUI
import AVFoundation


struct ContentView: View {
    @StateObject var cubeFaces = CubeFaces()
    
    let COLOUR: [Color] = [.blue, .green, .orange, .red, .yellow, .white]
    let HEIGHT: CGFloat = 266.66
    
    var body: some View {
        ZStack(alignment: .bottom) {
            StreamView(cubeFaces: cubeFaces)
                .ignoresSafeArea()
            
            ZStack(alignment: .bottom) {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Material.thin)
                    .frame(height: HEIGHT)
                
                Canvas { context, size in
                    let cubeSize = size.width / 12
                    
                    // middle
                    for i in 0 ..< 4 {
                        for j in 0 ..< 3 {
                            for k in 0 ..< 3 {
                                let loc = CGPoint(x: CGFloat(i * 3) * cubeSize + CGFloat(k) * cubeSize, y: CGFloat(j) * cubeSize + CGFloat(size.height / 2) - CGFloat(1.5 * cubeSize))
                                let finalSize = CGSize(width: cubeSize, height: cubeSize)
                                let rect = CGRect(origin: loc, size: finalSize)
                                var path = Rectangle().path(in: rect)
                                let index: Int = i * 9 + j * 3 + k
                                print(index)
                                print(Int(self.cubeFaces.faces[index]))
                                context.fill(path, with: .color(COLOUR[Int(self.cubeFaces.faces[index])]))
                            }
                        }
                    }
                    
                    // top
                    for j in 0 ..< 3 {
                        for k in 0 ..< 3 {
                            let loc = CGPoint(x: 3 * cubeSize + CGFloat(k) * cubeSize, y: CGFloat(j) * cubeSize)
                            let finalSize = CGSize(width: cubeSize, height: cubeSize)
                            let rect = CGRect(origin: loc, size: finalSize)
                            var path = Rectangle().path(in: rect)
                            let index: Int = 4 * 9 + j * 3 + k
                            context.fill(path, with: .color(COLOUR[Int(self.cubeFaces.faces[index])]))
                        }
                    }
                    
                    // bottom
                    for j in 0 ..< 3 {
                        for k in 0 ..< 3 {
                            let loc = CGPoint(x: 3 * cubeSize + CGFloat(k) * cubeSize, y: CGFloat(j) * cubeSize + size.height - (3 * cubeSize))
                            let finalSize = CGSize(width: cubeSize, height: cubeSize)
                            let rect = CGRect(origin: loc, size: finalSize)
                            var path = Rectangle().path(in: rect)
                            let index: Int = 5 * 9 + j * 3 + k
                            context.fill(path, with: .color((COLOUR[Int(self.cubeFaces.faces[index])])))
                        }
                    }
                    
                }
                .frame(height: HEIGHT)
                .scaleEffect(0.9)
            }
            .padding()
            .frame(height: HEIGHT)
        }
    }
}

//#Preview {
//    ContentView()
//}
