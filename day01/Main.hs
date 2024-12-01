module Main (main) where

import Data.List (sort)
import Lib (readFileFromArgs)

main :: IO ()
main = do
  contents <- readFileFromArgs
  let (xs, ys) = unzip (readPairs contents)
  let (xs', ys') = (sort xs, sort ys)
  let totalDistance = sum (zipWith (\x y -> abs (x - y)) xs' ys')
  print totalDistance

readPairs :: String -> [(Int, Int)]
readPairs = map readPair . lines

readPair :: String -> (Int, Int)
readPair line = case words line of
  [x, y] -> (read x, read y)
  _ -> error "Invalid input"
