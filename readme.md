## Inspiration

All the existing apps require you to frame the cube specifically in a grid, taking a photo per side, with strict requirements on the orientation of the faces to be scanned in. Although inconvenient, no one has successfully achieved a cube state reconstruction through a quick, dynamic video scan of the cube. As two of our team members have a long standing interest in speedcubing, we thought that this was the perfect opportunity to develop such an application.

## What it does

The app allows the user to scan a Rubik's cube, by rotating the cube in front of the camera (like FaceID!) to show all 6 sides. It will then produce an exact colour map of the cube state, taking into account the many possibilities of the orientation of the cube and possible obstructions.

## How we built it

We built the app with Swift (SwiftUI + UIKit), which interfaces with the backend via C++ OpenCV. We initially wrote the OpenCV detection using Python, and later ported it over to C++ for Swift interoperability. We use a variety of classical CV techniques such as image masking, Gaussian blurring, and contour maps. 

## Challenges we ran into

This was a really hard problem! We found countless blog posts with unfinished solutions, and many forum posts about this, but no one had successfully solved it. Speedcubes now are almost entirely stickerless (with no borders), so we could not just run a simple parallelogram detection to obtain the colours. Additionally, we needed to create a robust model that could detect live movement through a video stream, while accounting for shadows, perspective, glare, and obstructions. As always, working with C++ was painful, and especially getting C++ to work with Swift was an incredible, but fun, challenge.

## Accomplishments that we're proud of

It works! It's able to pinpoint colours onto the cube, and create a map of the colours, updating live. We were able to achieve this through entirely classical computer vision techniques, no ML / neural networks. Our inference time is extremely fast, and the results are highly predictable, meaning that we will be able to make meaningful changes in the future as we continue to improve our model.

## What we learned

There's a reason why there are no solutions online :(
We initially way underestimated how long this would take, but were very surprised at our results and hope to continue working on it in the future.

## What's next for CubeCV

The original idea was to implement an AR cube solver / helper, where the app would scan the cube and show live, augmented reality shading on the cube to suggest moves to perform. We're excited to work on this after the hackathon, and potentially make this a real app (the first of its kind)!
