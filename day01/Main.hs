module Main (main) where

import Data.List (sort)
import Lib (readFileFromArgs)

main :: IO ()
main = do
  contents <- readFileFromArgs
  let (xs, ys) = unzip $ readPairs contents
  part1 (xs, ys)
  part2 (xs, ys)

part1 :: ([Int], [Int]) -> IO ()
part1 (xs, ys) = do
  let (xs', ys') = (sort xs, sort ys)
  let totalDistance = sum $ zipWith distance xs' ys'
  putStrLn $ "Part 1: " ++ show totalDistance

part2 :: ([Int], [Int]) -> IO ()
part2 (xs, ys) = do
  let total = sum $ map (\x -> x * countOccurrences x ys) xs
  putStrLn $ "Part 2: " ++ show total

readPairs :: String -> [(Int, Int)]
readPairs = map readPair . lines

readPair :: String -> (Int, Int)
readPair line = case words line of
  [x, y] -> (read x, read y)
  _ -> error "Invalid input"

distance :: Num a => a -> a -> a
distance x y = abs (x - y)

countOccurrences :: Eq a => a -> [a] -> Int
countOccurrences x = length . filter (== x)