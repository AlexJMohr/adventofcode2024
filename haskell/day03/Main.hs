module Main (main) where

import Data.Maybe (mapMaybe)
import Lib (readFileFromArgs)
import Text.Read (readMaybe)
import Text.Regex.TDFA ((=~))

main :: IO ()
main = do
  contents <- readFileFromArgs
  let muls = readMuls contents
  let total = sum $ map (uncurry (*)) muls
  putStrLn $ "Part 1: " ++ show total

getAllMatches :: String -> [[String]]
getAllMatches = (=~ "mul\\(([0-9]+),([0-9]+)\\)")

readMuls :: String -> [(Int, Int)]
readMuls = mapMaybe parseMul . getAllMatches

parseMul :: [String] -> Maybe (Int, Int)
parseMul [_, xStr, yStr] = (,) <$> readMaybe xStr <*> readMaybe yStr
parseMul _ = Nothing