module Main (main) where

import qualified Data.List as List
import Lib (readFileFromArgs)

main :: IO ()
main = do
  contents <- readFileFromArgs
  let (xs, ys) = unzip (readPairs contents)
  let (xs', ys') = (List.sort xs, List.sort ys)
  let totalDistance = sum (zipWith distance xs' ys')
  print totalDistance

readPairs :: String -> [(Int, Int)]
readPairs = map readPair . lines

readPair :: String -> (Int, Int)
readPair line = case words line of
  [x, y] -> (read x, read y)
  _ -> error "Invalid input"

distance :: Int -> Int -> Int
distance x y = abs (x - y)
