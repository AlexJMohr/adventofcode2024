module Main (main) where

import Lib (distance, readFileFromArgs)

main :: IO ()
main = do
  contents <- readFileFromArgs
  let reports = readReports contents
  part1 reports
  part2 reports

part1 :: [[Int]] -> IO ()
part1 reports = do
  let safeReports = filter isSafe reports
  putStrLn $ "Part 1: " ++ show (length safeReports)

part2 :: [[Int]] -> IO ()
part2 reports = do
  let safeReports = filter isSafeWithRemoval reports
  putStrLn $ "Part 2: " ++ show (length safeReports)

readReport :: String -> [Int]
readReport = map read . words

readReports :: String -> [[Int]]
readReports = map readReport . lines

isSafe :: [Int] -> Bool
isSafe [] = True
isSafe [_] = True
isSafe (x : y : xs)
  | x == y = False
  | otherwise = isSafeImpl (x : y : xs) (x `compare` y)

isSafeImpl :: [Int] -> Ordering -> Bool
isSafeImpl [] _ = True
isSafeImpl [_] _ = True
isSafeImpl (x : y : xs) ord = checkDistance x y && x `compare` y == ord && isSafeImpl (y : xs) ord

between :: Ord a => a -> a -> a -> Bool
between lo hi z = lo <= z && z <= hi

checkDistance :: Int -> Int -> Bool
checkDistance = (between 1 3 .) . distance

generateListsWithOneRemoved :: [a] -> [[a]]
generateListsWithOneRemoved [] = []
generateListsWithOneRemoved (x : xs) = xs : map (x :) (generateListsWithOneRemoved xs)

isSafeWithRemoval :: [Int] -> Bool
isSafeWithRemoval = any isSafe . generateListsWithOneRemoved
