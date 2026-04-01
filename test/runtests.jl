using Test
using TestItemRunner

@testsnippet ChemBERTASetup begin
    using ChemBERTa, DelimitedFiles, PythonCall
    CB_PyExt = Base.get_extension(ChemBERTa, :PythonCallExt)
    if Sys.islinux()
        using RDKitMinimalLib
        CB_RDKExt = Base.get_extension(ChemBERTa, :RDKitMinimalLibExt)
    end
end

@run_package_tests